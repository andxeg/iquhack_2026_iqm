"""
layout_selection.py

Utility functions to select a high-fidelity subset of IQM qubits
and return a Qiskit-compatible initial_layout.

Author: you
"""

from typing import Dict, List, Tuple, Optional


# =========================
# CALIBRATION DATA
# =========================

READOUT_P10 = {
    'QB1': 0.0125, 'QB2': 0.0060, 'QB3': 0.0100, 'QB4': 0.0120, 'QB5': 0.0125,
    'QB6': 0.0100, 'QB7': 0.0405, 'QB8': 0.0065, 'QB9': 0.0185, 'QB10': 0.0170,
    'QB11': 0.0095, 'QB12': 0.0140, 'QB13': 0.0180, 'QB14': 0.0160, 'QB15': 0.0125,
    'QB16': 0.0040, 'QB17': 0.0115, 'QB18': 0.0105, 'QB19': 0.0165, 'QB20': 0.0225,
    'QB21': 0.0100, 'QB22': 0.0205, 'QB23': 0.0065, 'QB24': 0.0135, 'QB25': 0.0195,
    'QB26': 0.0080, 'QB27': 0.0085, 'QB28': 0.0105, 'QB29': 0.0280, 'QB30': 0.0175,
    'QB31': 0.0730, 'QB32': 0.0220, 'QB33': 0.0065, 'QB34': 0.0090, 'QB35': 0.0210,
    'QB36': 0.0115, 'QB37': 0.0145, 'QB38': 0.0170, 'QB39': 0.0245, 'QB40': 0.0195,
    'QB41': 0.0185, 'QB42': 0.0235, 'QB43': 0.0165, 'QB44': 0.0150, 'QB45': 0.0290,
    'QB46': 0.0295, 'QB47': 0.0215, 'QB48': 0.0145, 'QB49': 0.0150, 'QB50': 0.0195,
    'QB51': 0.0180, 'QB52': 0.0185, 'QB53': 0.0145, 'QB54': 0.0175
}

READOUT_P01 = {
    'QB1': 0.0230, 'QB2': 0.0190, 'QB3': 0.0240, 'QB4': 0.1585, 'QB5': 0.0425,
    'QB6': 0.0105, 'QB7': 0.0505, 'QB8': 0.0160, 'QB9': 0.0235, 'QB10': 0.0275,
    'QB11': 0.0230, 'QB12': 0.0255, 'QB13': 0.0190, 'QB14': 0.0210, 'QB15': 0.0290,
    'QB16': 0.0135, 'QB17': 0.0275, 'QB18': 0.0115, 'QB19': 0.0205, 'QB20': 0.0200,
    'QB21': 0.0155, 'QB22': 0.0175, 'QB23': 0.0130, 'QB24': 0.0260, 'QB25': 0.0220,
    'QB26': 0.0195, 'QB27': 0.0190, 'QB28': 0.0730, 'QB29': 0.0690, 'QB30': 0.0290,
    'QB31': 0.0400, 'QB32': 0.0240, 'QB33': 0.0220, 'QB34': 0.0155, 'QB35': 0.0300,
    'QB36': 0.0135, 'QB37': 0.0400, 'QB38': 0.0255, 'QB39': 0.0270, 'QB40': 0.0395,
    'QB41': 0.0175, 'QB42': 0.0320, 'QB43': 0.0145, 'QB44': 0.0985, 'QB45': 0.0195,
    'QB46': 0.0315, 'QB47': 0.0445, 'QB48': 0.0200, 'QB49': 0.0160, 'QB50': 0.0205,
    'QB51': 0.0815, 'QB52': 0.0315, 'QB53': 0.0250, 'QB54': 0.0625
}


# =========================
# SCORING HELPERS
# =========================

def readout_error(qb: str) -> float:
    """Total readout error P(1|0)+P(0|1)."""
    return READOUT_P10[qb] + READOUT_P01[qb]


def qb_to_index(qb: str) -> int:
    """Convert 'QB7' → 6."""
    return int(qb.replace("QB", "")) - 1


# =========================
# PUBLIC API
# =========================

def build_initial_layout_list(num_logical_qubits: int, *, excluded_qubits=None):
    excluded_qubits = set(excluded_qubits or [])
    candidates = [qb for qb in READOUT_P10.keys() if qb not in excluded_qubits]
    ranked = sorted(candidates, key=readout_error)

    if len(ranked) < num_logical_qubits:
        raise ValueError("Not enough qubits after exclusions.")

    chosen = ranked[:num_logical_qubits]
    physical_indices = [qb_to_index(qb) for qb in chosen]

    # Qiskit-friendly: list where position i is the physical qubit for logical qubit i
    return physical_indices