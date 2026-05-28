"""
Scalability experiment: CCO vs. TSSP across community sizes.

Run from the repository root:
    python src/escalabilidad_test.py

Outputs:
    data/processed/Results_Complete_Comparison.csv
    results/figures/Figure_1_Solver_Time_Trend.png
    results/figures/Figure_2_Computational_Breakdown.png
"""

import os
import sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import SolverFactory

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CCO_Model_PVandPL_BESS_P2P import P2P_Model
from stochastic_2stage_model_VFG import P2P_Model_Stochastic

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_OUT    = os.path.join(ROOT_DIR, 'data', 'processed')
FIGURES_OUT = os.path.join(ROOT_DIR, 'results', 'figures')

os.makedirs(DATA_OUT,    exist_ok=True)
os.makedirs(FIGURES_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
tamanios_comunidad = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 40, 55]
Z_90 = 1.282   # Phi^{-1}(0.90): 90% confidence level (eta = 0.10)

ruta_excel_cco = os.path.join(DATA_RAW, 'Case_Study_10.xlsx')

resultados_detallados = []

# ---------------------------------------------------------------------------
# 1. Load data once for both models
# ---------------------------------------------------------------------------
print("Loading Excel data once for both models...")

modelo_cco_base = P2P_Model(phi_inv=Z_90)
t0 = time.time()
modelo_cco_base.ReadExcelFile(ruta_excel_cco)
t_lectura_cco = time.time() - t0
agentes_completos_cco = modelo_cco_base.Agents.copy()

modelo_tssp_base = P2P_Model_Stochastic()
t1 = time.time()
modelo_tssp_base.ReadExcelFiles()
t_lectura_tssp = time.time() - t1
agentes_completos_tssp = modelo_tssp_base.Agents.copy()

print(f"Data loaded. (CCO: {t_lectura_cco:.2f}s, TSSP: {t_lectura_tssp:.2f}s)")

# ---------------------------------------------------------------------------
# 2. Scalability sweep
# ---------------------------------------------------------------------------
for n_agentes in tamanios_comunidad:
    print(f"\n{'='*60}")
    print(f"  SCALABILITY TEST — {n_agentes} AGENTS")
    print(f"{'='*60}")

    # --- CCO ---
    print(f"\n[CCO] Building and solving for {n_agentes} agents...")
    modelo_cco_base.Agents = agentes_completos_cco[:n_agentes]
    modelo_cco_base.nA = n_agentes

    t_c_start = time.time()
    instancia_cco = modelo_cco_base.Model()
    t_construccion_cco = time.time() - t_c_start

    opt_cco = SolverFactory('gurobi')
    t_g_start = time.time()
    opt_cco.solve(instancia_cco, tee=False)
    t_gurobi_cco = time.time() - t_g_start

    tiempo_total_cco = t_construccion_cco + t_gurobi_cco
    print(f"[CCO] Construction: {t_construccion_cco:.2f}s | Solver: {t_gurobi_cco:.2f}s")

    # --- TSSP ---
    print(f"\n[TSSP] Building and solving for {n_agentes} agents...")
    modelo_tssp_base.Agents = agentes_completos_tssp[:n_agentes]
    modelo_tssp_base.nA = n_agentes

    t_c_start = time.time()
    instancia_tssp = modelo_tssp_base.Model()
    t_construccion_tssp = time.time() - t_c_start

    opt_tssp = SolverFactory('gurobi')
    t_g_start = time.time()
    opt_tssp.solve(instancia_tssp, tee=False)
    t_gurobi_tssp = time.time() - t_g_start

    tiempo_total_tssp = t_construccion_tssp + t_gurobi_tssp
    print(f"[TSSP] Construction: {t_construccion_tssp:.2f}s | Solver: {t_gurobi_tssp:.2f}s")

    resultados_detallados.append({
        'Num_Agents':                    n_agentes,
        'CCO_Model_Construction_Time_s': round(t_construccion_cco,  4),
        'CCO_Solver_Time_s':             round(t_gurobi_cco,         4),
        'CCO_Total_Time_s':              round(tiempo_total_cco,     4),
        'TSSP_Model_Construction_Time_s':round(t_construccion_tssp, 4),
        'TSSP_Solver_Time_s':            round(t_gurobi_tssp,        4),
        'TSSP_Total_Time_s':             round(tiempo_total_tssp,    4),
    })

# ---------------------------------------------------------------------------
# 3. Save results
# ---------------------------------------------------------------------------
df = pd.DataFrame(resultados_detallados)
csv_path = os.path.join(DATA_OUT, 'Results_Complete_Comparison.csv')
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
print("Generating figures...")

# Figure 1 — Solver time trend (linear scale)
plt.figure(figsize=(10, 6))
plt.plot(df['Num_Agents'], df['CCO_Total_Time_s'],
         marker='o', linewidth=2.5, label='CCO (Total Time)', color='#1f77b4')
plt.plot(df['Num_Agents'], df['TSSP_Total_Time_s'],
         marker='s', linewidth=2.5, label='TSSP (Total Time)', color='#d62728')
plt.xlabel(r'Number of Agents ($|\mathcal{A}|$)', fontsize=12)
plt.ylabel('Total Time [s]', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(tamanios_comunidad, rotation=45)
plt.tight_layout()
fig1_path = os.path.join(FIGURES_OUT, 'Figure_1_Solver_Time_Trend.png')
plt.savefig(fig1_path, dpi=300)
plt.close()

# Figure 2 — Stacked bar: construction vs. solver time
fig, ax = plt.subplots(figsize=(12, 7))
x     = np.arange(len(df['Num_Agents']))
width = 0.35

ax.bar(x - width/2, df['CCO_Model_Construction_Time_s'],  width,
       label='CCO: Construction (Pyomo)', color='#aec7e8', edgecolor='black')
ax.bar(x - width/2, df['CCO_Solver_Time_s'], width,
       bottom=df['CCO_Model_Construction_Time_s'],
       label='CCO: Solver (Gurobi)', color='#1f77b4', edgecolor='black')

ax.bar(x + width/2, df['TSSP_Model_Construction_Time_s'], width,
       label='TSSP: Construction (Pyomo)', color='#ff9896', edgecolor='black')
ax.bar(x + width/2, df['TSSP_Solver_Time_s'], width,
       bottom=df['TSSP_Model_Construction_Time_s'],
       label='TSSP: Solver (Gurobi)', color='#d62728', edgecolor='black')

ax.set_ylabel('Time [s]', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{n}' for n in df['Num_Agents']], fontsize=11)
ax.set_xlabel(r'Number of Agents ($|\mathcal{A}|$)', fontsize=12)
ax.legend(fontsize=11, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
fig2_path = os.path.join(FIGURES_OUT, 'Figure_2_Computational_Breakdown.png')
plt.savefig(fig2_path, dpi=300)
plt.close()

print(f"Figures saved to: {FIGURES_OUT}")
print("\n--- SUMMARY ---")
print(df.to_string(index=False))
