# zne_iqm_qiskit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli


# ----------------------------
# 1) State preparation
# ----------------------------

def make_state_circuit(
    n: int,
    kind: str = "ghz",
    *,
    statevector: Optional[Union[np.ndarray, Sequence[complex]]] = None,
    builder: Optional[Callable[[QuantumCircuit], None]] = None,
) -> QuantumCircuit:
    """
    Build an n-qubit circuit that prepares a state.

    Args:
        n: number of qubits
        kind: preset state: "ghz", "w" (small n), "random", "computational"
        statevector: explicit statevector (length 2**n). If provided, overrides `kind`.
        builder: custom function that takes a QuantumCircuit and appends gates.
                 If provided, overrides `kind` and `statevector`.

    Returns:
        QuantumCircuit with state prep (no measurements).
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    qc = QuantumCircuit(n)

    if builder is not None:
        builder(qc)
        return qc

    if statevector is not None:
        sv = np.asarray(statevector, dtype=complex)
        if sv.shape != (2**n,):
            raise ValueError(f"statevector must have shape ({2**n},), got {sv.shape}")
        # Normalize defensively
        norm = np.linalg.norm(sv)
        if not np.isclose(norm, 1.0):
            sv = sv / norm
        qc.initialize(sv, list(range(n)))
        return qc

    kind = kind.lower().strip()

    if kind == "ghz":
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        return qc

    if kind == "computational":
        # |0...0> (do nothing) – caller can add X gates via builder if desired
        return qc

    if kind == "random":
        # Haar-random pure state (via Statevector.random)
        sv = Statevector.random(2**n).data
        qc.initialize(sv, list(range(n)))
        return qc

    if kind == "w":
        # Simple W-state construction (works best for small n; for large n you may want a custom builder)
        # Prepares (|100..0> + |010..0> + ... + |0..001>)/sqrt(n)
        # One standard approach uses controlled rotations; keep it readable.
        if n == 1:
            qc.x(0)
            return qc
        # Start at |100..0>
        qc.x(0)
        for k in range(1, n):
            theta = 2 * np.arccos(np.sqrt((n - k) / (n - k + 1)))
            qc.cry(theta, 0, k)
            qc.cx(k, 0)
        qc.x(0)
        return qc

    raise ValueError(f"Unknown kind='{kind}'. Use 'ghz', 'w', 'random', or provide statevector/builder.")


# ----------------------------
# 2) Observables (Pauli ops)
# ----------------------------

def make_observables(n: int) -> Dict[str, SparsePauliOp]:
    """
    Example observable set for n qubits, returned as a dict of SparsePauliOp.

    You can replace/extend this to whatever your experiment needs.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    obs: Dict[str, SparsePauliOp] = {}

    # Single-qubit Z on each qubit
    for i in range(n):
        label = ["I"] * n
        label[n - 1 - i] = "Z"  # Qiskit Pauli string order: leftmost acts on highest-index qubit
        obs[f"Z_{i}"] = SparsePauliOp("".join(label), coeffs=[1.0])

    # Global parity Z⊗Z⊗...⊗Z
    obs["Z_all"] = SparsePauliOp("Z" * n, coeffs=[1.0])

    # A couple of 2-local examples (if n>=2)
    if n >= 2:
        # Z0 Z1
        label = ["I"] * n
        label[n - 1 - 0] = "Z"
        label[n - 1 - 1] = "Z"
        obs["Z0Z1"] = SparsePauliOp("".join(label), coeffs=[1.0])

        # X0 X1
        label = ["I"] * n
        label[n - 1 - 0] = "X"
        label[n - 1 - 1] = "X"
        obs["X0X1"] = SparsePauliOp("".join(label), coeffs=[1.0])

    return obs


# ----------------------------
# Helpers: measurement + expectation
# ----------------------------

def _basis_change_for_pauli(pauli: Pauli) -> List[Tuple[str, int]]:
    """
    Returns a list of basis-change operations (gate name, qubit) to measure `pauli` in Z basis.
    For each qubit:
      X -> H
      Y -> Sdg then H
      Z/I -> nothing
    """
    ops = []
    # Pauli.to_label() uses 'IXYZ' ordering over qubits from high->low
    label = pauli.to_label()
    n = len(label)
    # Map label position to qubit index: label[0] acts on qubit n-1, label[n-1] acts on qubit 0
    for pos, p in enumerate(label):
        q = (n - 1) - pos
        if p == "X":
            ops.append(("h", q))
        elif p == "Y":
            ops.append(("sdg", q))
            ops.append(("h", q))
    return ops


def make_measurement_circuit(state_circuit: QuantumCircuit, observable: SparsePauliOp) -> QuantumCircuit:
    """
    Build a circuit that prepares `state_circuit` then measures in the basis needed for `observable`.

    Note: This implementation supports observables that are a (possibly sparse) *sum of Pauli strings*,
          but it measures each Pauli term separately (simple & robust, not maximally efficient).
    """
    if observable.num_qubits != state_circuit.num_qubits:
        raise ValueError("Observable and circuit qubit counts must match.")

    n = state_circuit.num_qubits

    # We'll return a circuit that measures all qubits in Z-basis *after basis change*.
    # The expectation computation will be done from bitstrings.
    qc = QuantumCircuit(n, n)
    qc.compose(state_circuit, inplace=True)

    # For now, assume a single Pauli term (common for quick ZNE demos).
    # If you pass a sum, you should loop terms outside and combine expectations.
    if len(observable.paulis) != 1:
        raise ValueError(
            "For simplicity, pass a single-term SparsePauliOp here. "
            "If you have a sum, loop over terms and combine with coefficients."
        )

    pauli = observable.paulis[0]
    for gate, q in _basis_change_for_pauli(pauli):
        if gate == "h":
            qc.h(q)
        elif gate == "sdg":
            qc.sdg(q)

    qc.measure(range(n), range(n))
    return qc


def expectation_from_counts(counts, pauli) -> float:
    """
    Compute <pauli> from counts, assuming we measured qubit i -> classical bit i.
    Qiskit count strings show classical bit 0 as the RIGHTMOST character.
    """
    label = pauli.to_label()   # left char acts on qubit n-1, right char on qubit 0
    n = len(label)
    shots = sum(counts.values())
    if shots == 0:
        raise ValueError("Counts are empty.")

    exp = 0.0
    for bitstring, c in counts.items():
        # Reverse so index == classical bit index == qubit index (because we measured i -> i)
        bits = bitstring[::-1]   # bits[q] is measurement result for qubit q

        val = 1.0
        for pos, p in enumerate(label):
            if p == "I":
                continue
            q = (n - 1) - pos     # map Pauli label position -> qubit index
            measured_bit = int(bits[q])
            val *= (1.0 if measured_bit == 0 else -1.0)

        exp += val * c

    return exp / shots



# ----------------------------
# 3) Digital folding for ZNE (noise scaling)
# ----------------------------

def fold_circuit_global(circuit: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Global unitary folding: U -> U (U†U)^k to get odd integer scale factors 1,3,5,...

    Requirements:
      - `scale` must be an odd integer >= 1

    Measurements are preserved at the end: we fold only the unitary part before measurement.

    This is the simplest folding strategy that stays within "just Qiskit".
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd integer >= 1 (e.g., 1, 3, 5, ...)")

    # Split off measurements if present (we assume measurements are at the end, typical for this workflow)
    # For safety, we just treat the full circuit as unitary if no measurements exist.
    # If your circuit has mid-circuit measurement, you should design a different scaling method.
    if circuit.num_clbits > 0 and circuit.count_ops().get("measure", 0) > 0:
        # crude but effective: remove final measurements by building a copy without them
        unitary = QuantumCircuit(circuit.num_qubits)
        for inst, qargs, cargs in circuit.data:
            if inst.name == "measure":
                continue
            unitary.append(inst, qargs, cargs)
        meas_part = circuit.copy()
        # We'll rebuild measurement separately:
        meas = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        # Copy classical mapping exactly:
        for inst, qargs, cargs in circuit.data:
            if inst.name == "measure":
                meas.append(inst, qargs, cargs)
    else:
        unitary = circuit.copy()
        meas = None

    k = (scale - 1) // 2
    folded = QuantumCircuit(unitary.num_qubits, circuit.num_clbits if meas is not None else 0)
    folded.compose(unitary, inplace=True)
    if k > 0:
        u_dag = unitary.inverse()
        for _ in range(k):
            folded.compose(u_dag, inplace=True)
            folded.compose(unitary, inplace=True)

    if meas is not None:
        folded.compose(meas, inplace=True)

    return folded


# ----------------------------
# 4) Run on IQM hardware
# ----------------------------

from typing import Optional, Sequence, Union
from qiskit import QuantumCircuit, transpile

def run_on_iqm_hardware(
    circuits: Union[QuantumCircuit, Sequence[QuantumCircuit]],
    *,
    server_url: str,
    quantum_computer: str,
    token: str = "rkSL242xyAzLHqJQrxHvHs2nREy4AbLE2aQcsA3unuoBnBZTQNV6spN3ItLtE2jJ",
    backend_name: Optional[str] = None,
    shots: int = 4000,
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = 1,
    cco = None,
    initial_layout=None
):
    """
    Execute circuit(s) on IQM (Resonance) using IQMProvider(server_url, quantum_computer=..., token=...).

    Args:
        circuits: a single QuantumCircuit or a list/tuple of circuits
        server_url: e.g. "https://resonance.meetiqm.com"
        quantum_computer: e.g. "emerald"
        token: your Resonance token
        backend_name: optional, usually None (provider.get_backend() default)
        shots: number of shots
        optimization_level: Qiskit transpiler optimization level
        seed_transpiler: deterministic transpilation seed (optional)
    """
    from iqm.qiskit_iqm import IQMProvider  # type: ignore

    provider = IQMProvider(
        server_url,
        quantum_computer=quantum_computer,
        token=token,
    )

    backend = provider.get_backend(backend_name) if backend_name else provider.get_backend()

    if isinstance(circuits, QuantumCircuit):
        circuits_list = [circuits]
    else:
        circuits_list = list(circuits)
    print(f"From run_on_iqm_hardware: initial layout is {initial_layout}")
    tqc = transpile(
        circuits_list,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        initial_layout=initial_layout
    )

    # Important: specify shots (your snippet omitted it; default may vary)
    job = backend.run(tqc, shots=shots, circuit_compilation_options = cco)
    result = job.result()
    return backend, result


# ----------------------------
# 5) ZNE runner + fit
# ----------------------------

@dataclass
class ZNEResult:
    scales: List[int]
    noisy_values: List[float]
    zne_value: float
    fit_coeffs: np.ndarray  # highest power first (numpy.polyfit)
    fit_degree: int


def run_zne(
    state_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    *,
    server_url: str,
    quantum_computer: str,
    token: str = "rkSL242xyAzLHqJQrxHvHs2nREy4AbLE2aQcsA3unuoBnBZTQNV6spN3ItLtE2jJ",
    backend_name: Optional[str] = None,
    shots: int = 4000,
    scales: Sequence[int] = (1, 3, 5),
    fit_degree: int = 1,
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = 1,
) -> ZNEResult:
    """
    Run ZNE by:
      1) building a measurement circuit for the observable
      2) folding globally for each scale factor (odd integers)
      3) running on IQM
      4) extracting expectation values
      5) extrapolating to scale=0 with a polynomial fit

    Note: fit is done vs 'scale' and extrapolated to scale=0.
    """
    if len(observable.paulis) != 1:
        raise ValueError("Pass a single-term SparsePauliOp here. Loop terms if you need sums.")

    pauli = observable.paulis[0]

    meas_circuit = make_measurement_circuit(state_circuit, observable)
    folded_circuits = [fold_circuit_global(meas_circuit, int(s)) for s in scales]

    _, result = run_on_iqm_hardware(
        folded_circuits,
        server_url=server_url,
        quantum_computer=quantum_computer,
        token=token,
        backend_name=backend_name,
        shots=shots,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )

    noisy_vals: List[float] = []
    for i in range(len(folded_circuits)):
        counts = result.get_counts(i)
        noisy_vals.append(expectation_from_counts(counts, pauli))

    # Polynomial fit and extrapolate to scale=0
    x = np.array([int(s) for s in scales], dtype=float)
    y = np.array(noisy_vals, dtype=float)

    if fit_degree >= len(scales):
        raise ValueError("fit_degree must be < number of scale points.")

    coeffs = np.polyfit(x, y, deg=fit_degree)
    zne_value = float(np.polyval(coeffs, 0.0))

    return ZNEResult(
        scales=list(map(int, scales)),
        noisy_values=noisy_vals,
        zne_value=zne_value,
        fit_coeffs=coeffs,
        fit_degree=fit_degree,
    )


# ----------------------------
# 6) Plot ZNE
# ----------------------------

def plot_zne(z: ZNEResult, *, title: str = "Zero-Noise Extrapolation (ZNE)") -> None:
    x = np.array(z.scales, dtype=float)
    y = np.array(z.noisy_values, dtype=float)

    # Plot points
    plt.figure()
    plt.plot(x, y, marker="o", linestyle="")

    # Plot fit curve from 0 to max scale
    xs = np.linspace(0.0, float(max(z.scales)), 200)
    ys = np.polyval(z.fit_coeffs, xs)
    plt.plot(xs, ys)

    plt.axvline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Noise scale factor (folding)")
    plt.ylabel("Expectation value")
    plt.text(
        0.02,
        0.02,
        f"ZNE @ scale=0: {z.zne_value:.6f}\nfit degree: {z.fit_degree}",
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
    )
    plt.show()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    n = 5

    # 1) Build an n-qubit state
    state = make_state_circuit(n, kind="ghz")

    # 2) Pick an observable
    observables = make_observables(n)
    obs = observables["Z_all"]  # single Pauli term

    # 3) Run ZNE on IQM
    # server_url examples are in IQM docs; you must use your endpoint + auth method. :contentReference[oaicite:2]{index=2}
    SERVER_URL = "https://YOUR_IQM_SERVER_URL_HERE"
    BACKEND_NAME = None  # or e.g. "iqm_garnet" depending on your account

    z = run_zne(
        state,
        obs,
        server_url=SERVER_URL,
        backend_name=BACKEND_NAME,
        shots=4000,
        scales=(1, 3, 5),
        fit_degree=1,            # linear is the usual first pass
        optimization_level=1,
        seed_transpiler=1,
    )

    print("Noisy values:", list(zip(z.scales, z.noisy_values)))
    print("ZNE extrapolated value:", z.zne_value)

    # 4) Plot
    plot_zne(z, title=f"ZNE on IQM for {obs.paulis[0].to_label()}")
