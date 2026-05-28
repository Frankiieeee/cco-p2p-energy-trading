# Chance-Constrained Optimization for Peer-to-Peer Energy Trading in Active Distribution Networks

Source code and input data for the paper submitted to **Sustainable Energy, Grids and Networks (SEGAN), Elsevier**.

---

## Repository Structure

```
├── code/
│   ├── CCO_Model_PVandPL_BESS_P2P.py       # Chance-constrained P2P model (Pyomo/Gurobi)
│   ├── stochastic_2stage_model_VFG.py       # Two-stage stochastic benchmark (TSSP)
│   └── escalabilidad_test.py                # Scalability experiment (CCO vs. TSSP)
│
└── data/
    ├── Case_Study_05.xlsx                   # CCO input — forecast error ε = 5%
    ├── Case_Study_10.xlsx                   # CCO input — forecast error ε = 10%
    ├── Case_Study_15.xlsx                   # CCO input — forecast error ε = 15%
    ├── Case_Study_TSSP.xlsx                 # TSSP benchmark input (9 scenarios)
    └── Case_Study_Deterministic.xlsx        # Deterministic reference case
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

**CCO model (base case):**
```python
from code.CCO_Model_PVandPL_BESS_P2P import P2P_Model

model = P2P_Model(phi_inv=1.282, num_agents=10)   # phi_inv = Φ⁻¹(0.90)
model.ReadExcelFile('data/Case_Study_10.xlsx')
model.Solver()
```

**Scalability experiment:**
```bash
python code/escalabilidad_test.py
```

---

## Citation

> Cárdenas, F. et al. (2025). *Chance-Constrained Optimization for Peer-to-Peer Energy Trading with PV Generation and Battery Storage in Active Distribution Networks*. Sustainable Energy, Grids and Networks. *(Under review)*

---

## License

MIT License — see [LICENSE](LICENSE).
