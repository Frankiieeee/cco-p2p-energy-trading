# A Chance-Constrained Upstream Energy-Exchange Commitment Model for Local Energy Communities with Peer-to-Peer Trading Under Forecast Uncertainty

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Source code and data for the paper submitted to **Sustainable Energy, Grids and Networks (SEGAN), Elsevier**.

---

## Repository Structure

```
├── code/
│   ├── CCO_Model_PVandPL_BESS_P2P.py     # CCO model — sensitivity sweep (all ε and η)
│   ├── CCO_Model_Single_Instance.py      # CCO model — single configurable instance
│   ├── stochastic_2stage_model_VFG.py    # Two-stage stochastic benchmark (TSSP)
│   ├── escalabilidad_test.py             # Scalability experiment (CCO vs. TSSP)
│   └── Determ_365.py                     # Deterministic model — 365-day annual run
│
└── data/
    ├── Case_Study/
    │   ├── Case_Study_05.xlsx            # CCO input — forecast error ε = 5%
    │   ├── Case_Study_10.xlsx            # CCO input — forecast error ε = 10%
    │   ├── Case_Study_15.xlsx            # CCO input — forecast error ε = 15%
    │   ├── Case_Study_TSSP.xlsx          # TSSP benchmark input (9 scenarios)
    │   └── Case_Study_365.xlsx           # Deterministic model input (365 days)
    │
    └── Results/
        ├── Resultados_CCO/
        │   ├── Results_Error5_Conf{50,75,90,95,99}.xlsx   # CCO results — ε = 5%
        │   ├── Results_Error10_Conf{50,75,90,95,99}.xlsx  # CCO results — ε = 10%
        │   ├── Results_Error15_Conf{50,75,90,95,99}.xlsx  # CCO results — ε = 15%
        │   └── Resumen_Global_Sensibilidad_CCO.xlsx        # Sensitivity summary (all ε, η)
        └── Resultados_TSSP/
            ├── Results_First_Stage_TSSP.xlsx               # TSSP first-stage schedule
            └── Results_S_{1..9}.xlsx                       # TSSP recourse per scenario
```

---

## Requirements

```
python >= 3.9
pyomo >= 6.4
gurobipy >= 10.0
pandas >= 1.5
numpy >= 1.23
openpyxl >= 3.0
```

> A valid [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) is required. Free academic licenses are available.

---

## Usage

**CCO — single instance** (edit `excel_file`, `error_val`, `prob_val`, `z_val` at the top of `__main__`):
```bash
python code/CCO_Model_Single_Instance.py
```

**CCO sensitivity sweep (all ε and η combinations):**
```bash
python code/CCO_Model_PVandPL_BESS_P2P.py
```

**TSSP benchmark (single run):**
```bash
python code/stochastic_2stage_model_VFG.py
```

**Scalability experiment (CCO vs. TSSP, 5–55 agents):**
```bash
python code/escalabilidad_test.py
```

**Deterministic model (365-day annual horizon):**
```bash
python code/Determ_365.py
```

---

## Citation

> Cárdenas, F. et al. (2025). *A Chance-Constrained Upstream Energy-Exchange Commitment Model for Local Energy Communities with Peer-to-Peer Trading Under Forecast Uncertainty*. Sustainable Energy, Grids and Networks. *(Under review)*

---

## License

MIT — see [LICENSE](LICENSE).
