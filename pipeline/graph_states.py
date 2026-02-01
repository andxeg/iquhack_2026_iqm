from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


def build_graph_state(G):
    """
    Construct an n-qubit graph state |G⟩ from a classical graph G.

    Definition:
        - Each vertex ↔ one qubit
        - Each edge (u, v) ↔ a CZ gate between qubits u and v

    Construction:
        1. Prepare |+⟩^{⊗n} by applying H to every qubit
        2. Apply CZ on every edge of the graph

    The resulting state satisfies:
        K_i |G⟩ = |G⟩
    for all stabilizers:
        K_i = X_i ⊗ ∏_{j ∈ N(i)} Z_j

    Args:
        G (networkx.Graph): Graph defining the entanglement structure

    Returns:
        QuantumCircuit: state-preparation circuit with no measurements
    """
    n = G.number_of_nodes()
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for u, v in G.edges():
        qc.cz(u, v)
    return qc


        
def graph_stabilizers(n, edges):
    """
    Generate the stabilizer generators for an n-qubit graph state.

    For each qubit i, the stabilizer is:
        K_i = X_i ⊗ ∏_{j ∈ N(i)} Z_j

    These operators uniquely define the graph state |G⟩ as their
    simultaneous +1 eigenstate.

    Important:
        - Stabilizers are returned as Pauli *labels* in Qiskit ordering
        - Qiskit Pauli strings act left-to-right on qubits n-1 → 0

    Args:
        n (int): Number of qubits (graph vertices)
        edges (iterable): Edge list defining neighbor relations

    Returns:
        List[str]: Pauli labels for stabilizers (length = n)
    """
    neighbors = {i: [] for i in range(n)}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    stabilizers = []
    for i in range(n):
        pauli = ['I'] * n
        pauli[i] = 'X'
        for j in neighbors[i]:
            pauli[j] = 'Z'
        stabilizers.append("".join(pauli[::-1]))  # Qiskit endian convention
    return stabilizers


def expectation_from_counts(counts, pauli_label: str) -> float:
    """
    Compute ⟨P⟩ for a Pauli operator P from raw measurement counts.

    Assumptions:
        - Measurement is performed in the Z basis on all qubits
        - Basis rotations (H, S†H) have already been applied as needed
        - measure_all() maps qubit i → classical bit i
        - Qiskit count strings are ordered with classical bit 0 on the RIGHT

    Pauli label convention:
        - pauli_label[0] acts on qubit n-1
        - pauli_label[n-1] acts on qubit 0

    Method:
        - Each measurement outcome contributes ±1 depending on parity
        - Expectation is computed as a weighted average over shots

    Args:
        counts (dict): measurement results from Qiskit backend
        pauli_label (str): Pauli operator label (e.g. "ZIXYZ")

    Returns:
        float: expectation value ⟨P⟩
    """
    n = len(pauli_label)
    shots = sum(counts.values())
    exp = 0.0

    for bitstring, c in counts.items():
        bits = bitstring[::-1]  # bits[q] is result for qubit q
        eigen = 1.0
        for pos, p in enumerate(pauli_label):
            if p == "I":
                continue
            q = (n - 1) - pos
            if bits[q] == "1":
                eigen *= -1.0
        exp += eigen * c

    return exp / shots



def graph_witness(expectations):
    """
    Evaluate the canonical graph-state entanglement witness.

    Witness operator:
        W = (n - 1)/2 · I  −  (1/2) · Σ_i K_i

    Properties:
        - ⟨W⟩ ≥ 0 for all fully separable n-qubit states
        - ⟨W⟩ = -1/2 for the ideal graph state
        - ⟨W⟩ < 0 certifies genuine multipartite entanglement

    Args:
        expectations (List[float]): ⟨K_i⟩ for each stabilizer

    Returns:
        float: witness expectation value
    """
    n = len(expectations)
    return (n - 1)/2 - 0.5 * sum(expectations)



def stabilizer_measurement_circuit(state_circuit, pauli):
    """
    Construct a circuit to measure a single stabilizer K_i.

    Procedure:
        1. Prepare the state |ψ⟩ using `state_circuit`
        2. Apply basis rotations so that:
           - X → H
           - Y → S† then H
           - Z → no change
        3. Measure all qubits in the Z basis

    The resulting bitstrings allow reconstruction of ⟨K_i⟩.

    Args:
        state_circuit (QuantumCircuit): state-preparation circuit
        pauli (str): Pauli label for the stabilizer

    Returns:
        QuantumCircuit: measurement circuit
    """
    qc = state_circuit.copy()

    for i, p in enumerate(pauli[::-1]):
        if p == 'X':
            qc.h(i)
        elif p == 'Y':
            qc.sdg(i)
            qc.h(i)

    qc.measure_all()
    return qc




