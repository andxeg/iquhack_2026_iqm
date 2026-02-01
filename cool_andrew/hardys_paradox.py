import numpy as np

from qiskit import QuantumCircuit, transpile


def hardys_circuit(theta0, theta1):
    """
    q0 = "electron path"
    q1 = "positron path"
    q2 = annihilation flag ancilla
    Measure order -> classical bits [0,1,2] so output bitstring is usually "q2 q1 q0".
    """
    qc = QuantumCircuit(3, 3)

    # First "beam splitters"
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)

    # Mark annihilation if both are in the interacting arm
    qc.ccx(0, 1, 2)

    # Second "beam splitters"
    qc.ry(np.pi - theta0, 0)
    qc.ry(np.pi - theta1, 1)

    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def hardy_gamma_from_counts(counts):
    """
    gamma = P(q0=0,q1=0 | q2=0)
    With bitstrings typically "q2 q1 q0".
    """
    total_q2_0 = 0
    q0q1_00_and_q2_0 = 0

    for bitstring, ct in counts.items():
        # defensively strip spaces if any backend formats weirdly
        b = bitstring.replace(" ", "")
        if len(b) < 3:
            continue

        q2, q1, q0 = b[0], b[1], b[2]

        if q2 == "0":
            total_q2_0 += ct
            if q0 == "0" and q1 == "0":
                q0q1_00_and_q2_0 += ct

    return (q0q1_00_and_q2_0 / total_q2_0) if total_q2_0 else 0.0

