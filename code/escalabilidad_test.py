"""
Scalability experiment: CCO vs. TSSP across community sizes.

Run from the repository root:
    python code/escalabilidad_test.py

Outputs:
    results/Results_Complete_Comparison.csv
"""

import os, sys, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyomo.environ import SolverFactory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CCO_Model_PVandPL_BESS_P2P import P2P_Model
from stochastic_2stage_model_VFG import P2P_Model_Stochastic

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

COMMUNITY_SIZES = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 40, 55]
Z_90 = 1.282  # Phi^{-1}(0.90)

# --- Load data once ---
print("Loading data...")
cco_base = P2P_Model(phi_inv=Z_90)
cco_base.ReadExcelFile(os.path.join(DATA_DIR, 'Case_Study_10.xlsx'))
agents_cco = cco_base.Agents.copy()

tssp_base = P2P_Model_Stochastic()
tssp_base.ReadExcelFiles()
agents_tssp = tssp_base.Agents.copy()

# --- Sweep ---
rows = []
for n in COMMUNITY_SIZES:
    print(f"\n[{n} agents]")

    cco_base.Agents = agents_cco[:n]; cco_base.nA = n
    t0 = time.time(); inst = cco_base.Model(); t_build_cco = time.time() - t0
    t0 = time.time(); SolverFactory('gurobi').solve(inst, tee=False); t_solve_cco = time.time() - t0

    tssp_base.Agents = agents_tssp[:n]; tssp_base.nA = n
    t0 = time.time(); inst = tssp_base.Model(); t_build_tssp = time.time() - t0
    t0 = time.time(); SolverFactory('gurobi').solve(inst, tee=False); t_solve_tssp = time.time() - t0

    rows.append({
        'Num_Agents':                     n,
        'CCO_Model_Construction_Time_s':  round(t_build_cco,  4),
        'CCO_Solver_Time_s':              round(t_solve_cco,  4),
        'CCO_Total_Time_s':               round(t_build_cco + t_solve_cco, 4),
        'TSSP_Model_Construction_Time_s': round(t_build_tssp, 4),
        'TSSP_Solver_Time_s':             round(t_solve_tssp, 4),
        'TSSP_Total_Time_s':              round(t_build_tssp + t_solve_tssp, 4),
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, 'Results_Complete_Comparison.csv')
df.to_csv(csv_path, index=False)
print(f"\nResults saved: {csv_path}")
print(df.to_string(index=False))
