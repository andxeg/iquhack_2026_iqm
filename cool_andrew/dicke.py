from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

import numpy as np
import matplotlib.pyplot as plt

# main classes and functions
from qiskit import QuantumRegister, QuantumCircuit

# gates
from qiskit.circuit.library import RYGate
from typing import Optional
from zne_entanglement_iqm import run_on_iqm_hardware



##################################################################
##################################################################
# auxiliary functions
##################################################################
##################################################################

def show_figure(fig):
    '''
    auxiliar function to display plot 
    even if it's not the last command of the cell
    from: https://github.com/Qiskit/qiskit-terra/issues/1682
    '''
    
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)
    

##################################################################
##################################################################
# quantum circuit
##################################################################
##################################################################

def gate_i(n, draw=False):
    '''
    returns the 2-qubit gate (i), introduced in Sec. 2.2
    
    this gate is simply a cR_y between 2 CNOTs, with
    rotation angle given by 2*np.arccos(np.sqrt(1/n))
    
    args: 
        - `n`: int, defining the rotation angle;
        - `draw` : bool, determines if the circuit is drawn.
    '''
    
    qc_i = QuantumCircuit(2)

    qc_i.cx(0, 1)

    theta = 2*np.arccos(np.sqrt(1/n))
    cry = RYGate(theta).control(ctrl_state="1")
    qc_i.append(cry, [1, 0])

    qc_i.cx(0, 1)

    ####################################

    gate_i = qc_i.to_gate()

    gate_i.name = "$(i)$"

    ####################################

    if draw:
        show_figure(qc_i.draw("mpl"))
        
    return gate_i

# ===============================================

def gate_ii_l(l, n, draw=False):
    '''
    returns the 2-qubit gate (ii)_l, introduced in Sec. 2.2
    
    this gate is simply a ccR_y between 2 CNOTs, with
    rotation angle given by 2*np.arccos(np.sqrt(l/n))
    
    args: 
        - `l`: int, defining the rotation angle;
        - `n`: int, defining the rotation angle;
        - `draw` : bool, determines if the circuit is drawn.
    '''

    qc_ii = QuantumCircuit(3)

    qc_ii.cx(0, 2)

    theta = 2*np.arccos(np.sqrt(l/n))
    ccry = RYGate(theta).control(num_ctrl_qubits = 2, ctrl_state="11")
    qc_ii.append(ccry, [2, 1, 0])

    qc_ii.cx(0, 2)

    ####################################

    gate_ii = qc_ii.to_gate()

    gate_ii.name = f"$(ii)_{l}$"

    ####################################

    if draw:
        show_figure(qc_ii.draw("mpl"))
        
    return gate_ii

# ===============================================

def gate_scs_nk(n, k, draw=False):
    '''
    returns the SCS_{n,k} gate, which is introduced in Definition 3 and
    if built of one 2-qubits (i) gate and k+1 3-qubits gates (ii)_l
    
    args: 
        - `n`: int, used as argument of `gate_i()` and `gate_ii_l()`;
        - `k`: int, defines the size of the gate, as well as its topology;
        - `draw` : bool, determines if the circuit is drawn.
    '''
    
    qc_scs = QuantumCircuit(k+1)

    qc_scs.append(gate_i(n), [k-1, k])

    for l in range(2, k+1):

        qc_scs.append(gate_ii_l(l, n), [k-l, k-l+1, k])

    ####################################

    gate_scs = qc_scs.to_gate()

    gate_scs.name = "SCS$_{" + f"{n},{k}" + "}$"

    ####################################

    if draw:
        show_figure(qc_scs.decompose().draw("mpl"))

    return gate_scs

# ===============================================

def first_block(n, k, l, draw=False):
    '''
    returns the first block of unitaries products in Lemma 2.
    
    args:
        - `n`: int, used to calculate indices and as argument of `gate_scs_nk()`;
        - `k`: int, used to calculate indices and as argument of `gate_scs_nk()`;
        - `l`: int, used to calculate indices and as argument of `gate_scs_nk()`;
        - `draw` : bool, determines if the circuit is drawn.
    '''

    qr = QuantumRegister(n)
    qc_first_block = QuantumCircuit(qr)

    # indices of qubits to which we append identities.
    n_first = l-k-1
    n_last = n-l
    
    # this will be a list with the qubits indices to which
    # we append the SCS_{n,k} gate.
    idxs_scs = list(range(n))

    # only if ther is an identity to append to first registers
    if n_first != 0:
        
        # correcting the qubits indices to which we apply SCS_{n, k}
        idxs_scs = idxs_scs[n_first:]
        

    if n_last !=0:
        
        idxs_scs = idxs_scs[:-n_last]
    
        # appending SCS_{n, k} to appropriate indices
        qc_first_block.append(gate_scs_nk(l, k), idxs_scs)

    
    else:
        
        # appending SCS_{n, k} to appropriate indices
        qc_first_block.append(gate_scs_nk(l, k), idxs_scs)
        
    # string with the operator, to be used as gate label
    str_operator = "$1^{\otimes " + f'{n_first}' + "} \otimes$ SCS$_{" + f"{l},{k}" + "} \otimes 1^{\otimes " + f"{n_last}" + "}$"
    
    ####################################

    gate_first_block = qc_first_block.to_gate()

    gate_first_block.name = str_operator

    ####################################

    if draw:
        show_figure(qc_first_block.decompose().draw("mpl"))
        
    return gate_first_block

# ===============================================

def second_block(n, k, l, draw=False):    
    '''
    returns the second block of unitaries products in Lemma 2.
    
    args:
        - `n`: int, used to calculate indices and as argument of `gate_scs_nk()`;
        - `k`: int, used to calculate indices and as argument of `gate_scs_nk()`;
        - `l`: int, used to calculate indices and as argument of `gate_scs_nk()`;
        - `draw` : bool, determines if the circuit is drawn.
    '''
        
    qr = QuantumRegister(n)
    qc_second_block = QuantumCircuit(qr)

    # here we only have identities added to last registers, so it's a bit simpler
    n_last = n-l
    
    idxs_scs = list(range(n))

    if n_last !=0:
        
        idxs_scs = idxs_scs[:-n_last]
    
        # appending SCS_{n, k} to appropriate indices
        qc_second_block.append(gate_scs_nk(l, l-1), idxs_scs)
    
    else:
        
        # appending SCS_{n, k} to appropriate indices
        qc_second_block.append(gate_scs_nk(l, l-1), idxs_scs)
        
    str_operator = "SCS$_{" + f"{l},{l-1}" + "} \otimes 1^{\otimes " + f"{n_last}" + "}$"
    
    ####################################

    gate_second_block = qc_second_block.to_gate()

    gate_second_block.name = str_operator

    ####################################

    if draw:
        show_figure(qc_second_block.decompose().draw("mpl"))
        
    return gate_second_block

# ===============================================

def dicke_state(n, k, draw=False, barrier=False, only_decomposed=False):
    '''
    returns a quantum circuit with to prepare a Dicke state with arbitrary n, k
    
    this function essentially calls previous functions, wrapping the products in Lemma 2
    within the appropriate for loops for the construction of U_{n, k}, which is
    applied to the initial state as prescribed.
    
    args:
        - `n`: int, number of qubits for the Dicke state;
        - `k`: int, hamming weight of states to be included in the superposition;
        - `draw` : bool, determines if the circuit is drawn;
        - `barrier` : bool, determines if barriers are added to the circuit;
        - `only_decomposed` : bool, determines if only the 3-fold decomposed circuit is drawn.
        
    '''

    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    ########################################
    # initial state
    
    qc.x(qr[-k:])

    if barrier:
        
        qc.barrier()

    ########################################
    # applying U_{n, k}
    # I didn't create a function to generate a U_{n, k} gate
    # because I wanted to keep the possibility of adding barriers,
    # which are quite nice for visualizing the circuit and its elements.
    
    # apply first term in Lemma 2
    # notice how this range captures exactly the product bounds.
    # also, it's very important to invert the range, for the gates
    # to be applied in the correct order!!
    for l in range(k+1, n+1)[::-1]:

        qc.append(first_block(n, k, l), range(n))

        if barrier:
            
            qc.barrier()

    # apply second term in Lemma 2
    for l in range(2, k+1)[::-1]:

        qc.append(second_block(n, k, l), range(n))

        if barrier:
            
            qc.barrier()
      
    ########################################
    # drawing
    if draw:
        
        # draws only the 3-fold decomposed circuit. 
        # this opens up to (i) and (ii)_l into its components.
        if only_decomposed:
            
            show_figure(qc.decompose().decompose().decompose().draw("mpl"))
            
        else:
            
            show_figure(qc.draw("mpl"))
            print()
            show_figure(qc.decompose().draw("mpl"))
            print()
            show_figure(qc.decompose().decompose().draw("mpl"))
            print()
            show_figure(qc.decompose().decompose().decompose().draw("mpl"))
    
    return qc

def dicke_witness_threshold(N: int) -> float:
    # C = 1/2 * N/(N-1)  (for Dicke state with N/2 excitations, N even)
    return 0.5 * N / (N - 1)

def build_witness_circuit(prep: QuantumCircuit, k: int) -> QuantumCircuit:
    """
    prep: your N-qubit state preparation circuit U acting on |0...0>
    k: number of excitations in Dicke state |k, N>
       (for the bound cited, use k = N//2 and N even)
    Returns circuit that estimates F = |<D|psi>|^2 via P(0...0).
    """
    N = prep.num_qubits
    dicke = dicke_state(N, k)           # prepares |k, N> from |0...0>
    qc = QuantumCircuit(N, N)
    qc.compose(prep, inplace=True)
    qc.compose(dicke.inverse(), inplace=True)  # apply U_D^\dagger
    qc.measure(range(N), range(N))
    return qc

def estimate_fidelity_allzeros(qc: QuantumCircuit, shots: int = 20_000) -> float:
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    N = qc.num_qubits
    return counts.get("0"*N, 0) / shots

def dicke_witness_value(F: float, N: int) -> float:
    C = dicke_witness_threshold(N)
    return C - F  # <W> = C - F ; negative => GME detected

def estimate_fidelity_allzeros_from_result(result, N: int) -> float:
    """
    Works for IQM result object (and also Aer result) as long as result.get_counts() exists.
    """
    counts = result.get_counts()  # single-circuit case
    shots = sum(counts.values())
    return counts.get("0" * N, 0) / shots

def run_dicke_fidelity_witness_on_iqm(
    *,
    N: int,
    k: int,
    prep: QuantumCircuit,
    server_url: str,
    quantum_computer: str,
    backend_name: Optional[str] = None,
    shots: int = 4000,
    optimization_level: int = 1,
    seed_transpiler: Optional[int] = 1,
):
    """
    Runs the fidelity-style Dicke witness on IQM hardware.

    Witness circuit:  U  -> U_D^\dagger -> measure
      where U is `prep` (your actual prepared state circuit)
            U_D is `dicke_state(N,k)` (your Dicke prep from the file)

    Returns dict with counts, F_hat, C, W = C-F, and GME flag (only meaningful for balanced k=N/2, N even).
    """
    # Build witness/verification circuit using your existing helper
    witness_circ = build_witness_circuit(prep, k)

    backend, result = run_on_iqm_hardware(
        witness_circ,
        server_url=server_url,
        quantum_computer=quantum_computer,
        backend_name=backend_name,
        shots=shots,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )

    counts = result.get_counts()
    F_hat = counts.get("0" * N, 0) / sum(counts.values())

    # Only use the "GME threshold" you coded when it's the balanced Dicke case (k=N/2, N even).
    C = None
    W = None
    gme_detected = None
    if (N % 2 == 0) and (k == N // 2):
        C = dicke_witness_threshold(N)
        W = dicke_witness_value(F_hat, N)  # = C - F
        gme_detected = (W < 0)

    return {
        "backend_name": backend,
        "counts": counts,
        "F_hat": F_hat,
        "C_threshold": C,
        "W_exp": W,
        "gme_detected": gme_detected,
        "witness_circuit": witness_circ,
    }



