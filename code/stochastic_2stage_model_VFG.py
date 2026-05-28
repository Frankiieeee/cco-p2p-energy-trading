import numpy as np
import pandas as pd
from pyomo.environ import *
import pyomo.environ as py
import sys
import os
import time                 
import datetime             

class P2P_Model_Stochastic:
    def __init__(self, num_agents=55, name=None):
        self.data = {}
        self.Time = 24 
        self.refB = 1
        self.RR = 12
        self.MWtoKw = 1000
        self.sb = 100
        todos_los_agentes = [7,8,10,11,18,19,22,25,26,28,31,32,38,41,42,45,46,51,53,57,58,60,62,74,83,84,94,95,109,110,127,130,133,137,138,142,144,146,148,151,153,154,157,158,160,165,166,169,170,173,174,177,178,191,192]
        self.Agents = todos_los_agentes[:num_agents] # Corta la lista al tamaño deseado
        self.nA = len(self.Agents)
        self.BigM = 1000
        self.Fi = 0.9
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        self.PB = 0.0034
        self.n_scenarios = 9
        self.scenario_prob = None
        self.Ppv_base = None
        self.demand_profiles_base = None
        
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.CASE_STUDY_FILE = os.path.join(_root, 'data', 'Case_Study_TSSP.xlsx')

    def ReadExcelFiles(self):
        try:
            if not os.path.exists(self.CASE_STUDY_FILE):
                raise FileNotFoundError(f"No se encontró el archivo: {self.CASE_STUDY_FILE}")

            self.Links = pd.read_excel(self.CASE_STUDY_FILE, sheet_name='Adj_Matrix')
            self.Buses = pd.read_excel(self.CASE_STUDY_FILE, sheet_name='Buses')
            self.Lines = pd.read_excel(self.CASE_STUDY_FILE, sheet_name='Lines')
            self.Scenarios = pd.read_excel(self.CASE_STUDY_FILE, sheet_name='S_Profiles')
            self.Circum = pd.read_excel(self.CASE_STUDY_FILE, sheet_name='Circum')
            self.scenario_prob = pd.read_excel(self.CASE_STUDY_FILE, sheet_name='Prob_Sc')
      
            if self.Buses.columns[0].startswith('Unnamed'):
                 self.Buses = pd.read_excel(self.CASE_STUDY_FILE, sheet_name='Buses', index_col=0)
            else:
                 self.Buses = self.Buses.set_index(self.Buses.columns[0])

            self.buses = len(self.Buses)
            self.x_matrix = pd.DataFrame(np.zeros([self.buses,self.buses])) 
            self.r_matrix = pd.DataFrame(np.zeros([self.buses,self.buses]))
            self.Smax = pd.DataFrame(np.zeros([self.buses,self.buses]))    
        
            for i in range(len(self.Lines)): 
                fbus = self.Lines.loc[i,'fbus'] 
                tbus = self.Lines.loc[i,'tbus'] 
                self.x_matrix.loc[fbus-1,tbus-1] = self.Lines.loc[i,'x'] 
                self.x_matrix.loc[tbus-1,fbus-1] = self.Lines.loc[i,'x']
                self.r_matrix.loc[fbus-1,tbus-1] = self.Lines.loc[i,'r'] 
                self.r_matrix.loc[tbus-1,fbus-1] = self.Lines.loc[i,'r']
                self.Smax.loc[fbus-1,tbus-1] = self.Lines.loc[i,'Smax']  
                self.Smax.loc[tbus-1,fbus-1] = self.Lines.loc[i,'Smax']
            
            print("Datos Excel cargados correctamente.")

        except FileNotFoundError as e:
            print(f"--- ERROR CRÍTICO ---")
            print(f"{e}")
            sys.exit()
        except ValueError as e:
            if "Worksheet" in str(e):
                print(f"--- ERROR CRÍTICO ---")
                print(f"No se encontró una hoja requerida en '{self.CASE_STUDY_FILE}'. {e}")
                sys.exit()
            else:
                raise e
        except KeyError as e:
            print(f"Error: No se encontró la columna {e} en uno de los archivos Excel.")
            sys.exit()


    
    def Model(self):
        model = AbstractModel()

        model.B = py.Set(initialize=np.arange(1,self.buses+1)) 
        model.A = py.Set(initialize=self.Agents)              
        model.T = py.Set(initialize=np.arange(1,self.Time+1))
        model.RR = py.Set(initialize=np.arange(1,self.RR+1))
        model.S = py.Set(initialize=np.arange(1, self.n_scenarios + 1)) # n_scenarios ahora es 12
        
        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.iloc[i,j]==1 if i<j)
        model.L = py.Set(dimen=2,initialize=Set_L_init)


        def PV_scenario_init(model, s, i, t):
            return (self.Buses.loc[i, 'DG']/ model.sb)*self.Scenarios.loc[t-1,'PV_'+str(s)]
        model.PV_s = py.Param(model.S, model.A, model.T, rule=PV_scenario_init)

        def PL_scenario_init(model, s, i, t):
            return (self.Buses.loc[i, 'PL']/model.sb)*self.Scenarios.loc[t-1,'P_'+str(i)+'_'+str(s)]
        model.pl = py.Param(model.S, model.A, model.T, rule=PL_scenario_init)

        def Prob_init(model, s):
            return self.scenario_prob.loc[0,'S_'+str(s)]
        model.prob = py.Param(model.S, rule=Prob_init)
        
        def QL_init(model,i,t):   
            if i in model.A:
                return self.Buses.loc[i, 'QL'] * self.Scenarios.loc[t-1,'P_'+str(i)+'_1']
            else:
                return 0
        model.QL = py.Param(model.B,model.T, rule=QL_init)

        def QGmin_init(model, i):
            return self.Buses.loc[i, 'QGmin'] / self.sb  
        model.QGmin = py.Param(model.B, rule=QGmin_init)

        def QGmax_init(model, i):
            return self.Buses.loc[i, 'QGmax'] / self.sb
        model.QGmax = py.Param(model.B, rule=QGmax_init)
        
        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]  
        model.X = py.Param(model.B,model.B,rule=x_init)
        
        def r_init(model,i,j):
            return self.r_matrix.loc[i-1,j-1]  
        model.R = py.Param(model.B,model.B,rule=r_init)
        
        def Price_init(model,t):
            return self.Scenarios.loc[t-1,'Price'] 
        model.Lambda_BG = py.Param(model.T, rule=Price_init)
        
        def Price_SG_init(model,t):
            return self.Scenarios.loc[t-1,'Price_SG'] 
        model.Lambda_SG = py.Param(model.T, rule=Price_SG_init)
        
        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]        
        model.s_max = py.Param(model.B,model.B,rule=Smax_init)
        
        def P_SG_max_init(model,i):
            return self.Buses['P_SG_max'].loc[i] 
        model.P_SG_max = py.Param(model.B, rule=P_SG_max_init)
        
        def P_BG_max_init(model,i):
            return self.Buses['P_BG_max'].loc[i] 
        model.P_BG_max = py.Param(model.B, rule=P_BG_max_init)
        
        def Vmax_init(model,i):
            return self.Buses['Vmax'].loc[i]  
        model.Vmax = py.Param(model.B, rule=Vmax_init)
        
        def Vmin_init(model,i):
            return self.Buses['Vmin'].loc[i] 
        model.Vmin = py.Param(model.B,  rule=Vmin_init)

        def AA_init(model,i):
            return self.Circum.loc[i-1,'AA']
        model.AA = py.Param(model.RR, rule=AA_init)

        def BB_init(model,i):
            return self.Circum.loc[i-1,'BB']
        model.BB = py.Param(model.RR, rule=BB_init)

        def CC_init(model,i):
            return self.Circum.loc[i-1,'CC']
        model.CC = py.Param(model.RR, rule=CC_init)

        def BT_init(model,i):
            return self.Buses.loc[i,'BT']  
        model.BT = py.Param(model.A, rule=BT_init)
        
        def E_BT_init(model,i):
            return self.Buses.loc[i,'E_BT']  
        model.E_BT = py.Param(model.A, rule=E_BT_init)

        model.Fi = py.Param(initialize = self.Fi) 
        model.SOC_min = self.SOC_min   
        model.SOC_max = self.SOC_max
        model.PB = self.PB        
        model.sb = 100  
        model.MWtoKW = 1000   
        model.nA = self.nA
        model.M = self.BigM
        
        model.y = py.Var(model.A, model.T, within=py.Binary, initialize=0)
        model.w = py.Var(model.A, model.T, within=py.Binary, initialize=0)

        model.pv = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0) 
        model.qg = py.Var(model.S, model.B, model.T, within=py.Reals, initialize=0)
        model.v = py.Var(model.S, model.B, model.T, within=py.NonNegativeReals, initialize=0)
        model.p = py.Var(model.S, model.L, model.T) 
        model.q = py.Var(model.S, model.L, model.T) 
        model.kappa_bg = py.Var(model.B, model.T, within=py.NonNegativeReals, initialize=0) 
        model.kappa_sg = py.Var(model.B, model.T, within=py.NonNegativeReals, initialize=0)

        model.dp = py.Var(model.S, model.A, model.T, within=py.Reals, initialize=0) 
        model.dp_pos = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0) 
        model.dp_neg = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0) 

        model.soc = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0)   
        model.ch = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0)    
        model.ds = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0)   

        model.p_sg = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.p_sm = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.p_bg = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.p_bm = py.Var(model.S, model.A, model.T, within=py.NonNegativeReals, initialize=0)

        def Fun_obj(model):
            first_stage = sum(model.Lambda_BG[t] * (model.kappa_bg[i,t] * model.sb) - model.Lambda_SG[t] * (model.kappa_sg[i,t] * model.sb) for i in model.B for t in model.T)
            second_stage = sum(model.prob[s] * sum(0.01 * (model.pv[s,i,t] + model.ds[s,i,t]) * model.sb for i in model.A for t in model.T) for s in model.S)
            return first_stage + second_stage 
        model.FunObj = py.Objective(rule=Fun_obj, sense=py.minimize)
        
        
        # --- RESTRICCIONES ---
        
        def NPFE_R_Agents(model, s, i, t):
            value1 = sum(model.p[s,m,n,t] for (m,n) in model.L if m==i) - sum(model.p[s,n,m,t] for (n,m) in model.L if m==i)
            if i in model.A:
                return model.dp[s,i,t] == value1
            else:
                return model.kappa_bg[i,t] - model.kappa_sg[i,t] == value1
        model.npfe_r_agents = py.Constraint(model.S, model.B, model.T, rule=NPFE_R_Agents)
        
        # def NPFE_R_Grid(model, s, i, t):
        #     if i not in model.A:
        #         value1 = sum(model.p[s,m,n,t] for (m,n) in model.L if m==i) - sum(model.p[s,n,m,t] for (n,m) in model.L if m==i)
        #         return model.kappa_bg[i,t] - model.kappa_sg[i,t] == value1
        #     else:
        #         return Constraint.Skip
        # model.npfe_r_grid = py.Constraint(model.S, model.B, model.T, rule=NPFE_R_Grid)

        def NPFE_Im(model, s, i, t):
            value2 = sum(model.q[s,m,n,t] for (m,n) in model.L if m==i) - sum(model.q[s,n,m,t] for (n,m) in model.L if m==i) 
            return model.qg[s,i,t] - model.QL[i,t] == value2
        model.npfe_im = py.Constraint(model.S, model.B, model.T, rule=NPFE_Im) 

        def LPFE_R(model, s, i, j, t):                                                
            return model.v[s,j,t] == model.v[s,i,t] - 2*(model.R[i,j]*model.p[s,i,j,t] + model.X[i,j]*model.q[s,i,j,t])
        model.lpfe_r = py.Constraint(model.S, model.L, model.T, rule=LPFE_R)

        def DP_init(model, s, i, t):
            return model.dp[s,i,t] == model.pv[s,i,t] - model.pl[s,i,t] + model.ds[s,i,t] - model.ch[s,i,t]  
        model.dp_init = py.Constraint(model.S, model.A, model.T, rule=DP_init)

        def DP_init2(model, s, i, t):
            return model.dp[s,i,t] == model.dp_pos[s,i,t] - model.dp_neg[s,i,t] 
        model.dp_init2 = py.Constraint(model.S, model.A, model.T, rule=DP_init2)

        def SOCR(model, s, i, t): 
            if t == 1:
                return model.soc[s,i,t] == 0.5 * (model.BT[i]/model.sb) #+ (model.Fi*model.ch[s,i,t]) - (model.ds[s,i,t]/model.Fi)
            else:
                return model.soc[s,i,t] == model.soc[s,i,t-1] + model.Fi*model.ch[s,i,t-1] - model.ds[s,i,t-1]/model.Fi
        model.socr = Constraint(model.S, model.A, model.T, rule=SOCR)

        def Limit_SOC(model, s, i, t): 
            return model.soc[s,i,t] >= (model.BT[i]/model.sb)*model.SOC_min
        model.limit_soc = Constraint(model.S, model.A, model.T, rule=Limit_SOC)

        def Limit_SOCsup(model, s, i, t):
            return model.soc[s,i,t] <= (model.BT[i]/model.sb)*model.SOC_max
        model.limit_socsup = Constraint(model.S, model.A, model.T, rule=Limit_SOCsup)

        def CHR(model, s, i, t): 
            return model.ch[s,i,t] <= (model.PB/model.sb) * model.w[i,t] 
        model.chr = Constraint(model.S, model.A, model.T, rule=CHR)

        def DSR(model, s, i, t): 
            return model.ds[s,i,t] <= (model.PB/model.sb)*(1-model.w[i,t])
        model.dsr = Constraint(model.S, model.A, model.T, rule=DSR)

        def DSR2(model, s, i, t): 
            return model.ds[s,i,t] <= model.soc[s,i,t]
        model.dsr2 = Constraint(model.S, model.A, model.T, rule=DSR2)

        def DSR_W(model, i, t): 
            return model.w[i,t] <= model.E_BT[i]
        model.dsr_w = Constraint(model.A, model.T, rule=DSR_W)

        def DP_Pos_2(model, s, i, t):
            return model.dp_pos[s,i,t] == model.p_sg[s,i,t] + model.p_sm[s,i,t] 
        model.DP_Pos_2 = py.Constraint(model.S, model.A, model.T, rule=DP_Pos_2)

        def DP_Neg_2(model, s, i, t):
            return model.dp_neg[s,i,t] == model.p_bg[s,i,t] + model.p_bm[s,i,t] 
        model.DP_Neg_2 = py.Constraint(model.S, model.A, model.T, rule=DP_Neg_2)

        def Market_Balance(model, s, t):
            return sum(model.p_sm[s,i,t] for i in model.A) == sum(model.p_bm[s,i,t] for i in model.A) 
        model.market_balance = py.Constraint(model.S, model.T, rule=Market_Balance)

        def DP_Pos(model, s, i, t):
            return model.dp_pos[s,i,t] <= model.M* model.y[i,t] 
        model.DP_Pos = py.Constraint(model.S, model.A, model.T, rule=DP_Pos)

        def DP_Neg(model, s, i, t):
            return model.dp_neg[s,i,t] <= model.M * (1 - model.y[i,t]) 
        model.DP_Neg = py.Constraint(model.S, model.A, model.T, rule=DP_Neg)

        def Kappa_BG_Balance(model, s, t):
            return sum(model.p_bg[s,i,t] for i in model.A) == sum(model.kappa_bg[i,t] for i in model.B) 
        model.kappa_bg_balance = py.Constraint(model.S, model.T, rule=Kappa_BG_Balance)

        def Kappa_SG_Balance(model, s, t):
            return sum(model.p_sg[s,i,t] for i in model.A) == sum(model.kappa_sg[i,t] for i in model.B) 
        model.kappa_sg_balance = py.Constraint(model.S, model.T, rule=Kappa_SG_Balance)

        def bus_uno_compra(model, i, t):
            return model.kappa_bg[i,t] <= model.P_BG_max[i]
        model.bus_uno_buy = Constraint(model.B, model.T, rule=bus_uno_compra)
        
        def bus_uno_venta(model, i, t):
            return model.kappa_sg[i,t] <= model.P_SG_max[i]
        model.bus_uno_sell = Constraint(model.B, model.T, rule=bus_uno_venta)

        def QG_max(model, s, i, t):
            return model.qg[s,i,t] <= model.QGmax[i]
        model.QG_max = py.Constraint(model.S, model.B, model.T, rule=QG_max)

        def QG_min(model, s, i, t):
            return model.qg[s,i,t] >= model.QGmin[i]
        model.QG_min = py.Constraint(model.S, model.B, model.T, rule=QG_min)

        def V_max(model, s, i, t):
            return model.v[s,i,t] <= model.Vmax[i]**2
        model.V_max = py.Constraint(model.S, model.B, model.T, rule=V_max)

        def V_min(model, s, i, t):
            return model.v[s,i,t] >= model.Vmin[i]**2
        model.V_min = py.Constraint(model.S, model.B, model.T, rule=V_min)

        def PV_gen(model, s, i, t):
            return model.pv[s,i,t] <= model.PV_s[s,i,t] 
        model.PV_gen = py.Constraint(model.S, model.A, model.T, rule=PV_gen)

        def Smax_constraint(model, s, i, j, t, r):
            return model.AA[r] * model.p[s,i,j,t] + model.BB[r] * model.q[s,i,j,t] + model.CC[r] * (model.s_max[i,j] / model.sb) <= 0
        model.Smax_constraint = py.Constraint(model.S, model.L, model.T, model.RR, rule=Smax_constraint)

      
        print("Creando instancia del modelo...")
        return model.create_instance()

    def Solver(self):
        self.ReadExcelFiles()
        Model = self.Model()
        
        self.opt = SolverFactory('gurobi') 
        print("\nIniciando optimización con Gurobi...")
        
        # Parámetros para ahorrar memoria
        # self.opt.options['Method'] = 2
        #self.opt.options['NodefileStart'] = 0.5 

        start_solve_time = time.time() 
        
        results_Model = self.opt.solve(Model, tee=True)
        results_Model.write()
        
        solve_duration = time.time() - start_solve_time 
        
        print(f"\n--- Optimización Finalizada. Tiempo de Gurobi: {solve_duration:.2f} segundos ---")
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        return Model

    def ExportResults(self, model):
        import os
        import pandas as pd
        from pyomo.environ import value

        base_path = os.getcwd()
        sb = self.sb
        MWtoKw = self.MWtoKw

        # ---------------------------------------------------------
        # 1. DEFINICIÓN DE VARIABLES
        # ---------------------------------------------------------

        # Variables dependientes del escenario (s, i, t) -> Results_S_{s}.xlsx
        vars_per_scenario = {
            'PV': lambda s, i, t: value(model.pv[s, i, t]) * MWtoKw * sb,
            'pl': lambda s, i, t: value(model.pl[s, i, t]) * MWtoKw * sb,
            'DP': lambda s, i, t: value(model.dp[s, i, t]) * MWtoKw * sb,
            'DP_pos': lambda s, i, t: value(model.dp_pos[s, i, t]) * MWtoKw * sb,
            'DP_neg': lambda s, i, t: value(model.dp_neg[s, i, t]) * MWtoKw * sb,
            'soc': lambda s, i, t: value(model.soc[s, i, t]) * MWtoKw * sb,
            'CH': lambda s, i, t: value(model.ch[s, i, t]) * MWtoKw * sb,
            'DS': lambda s, i, t: value(model.ds[s, i, t]) * MWtoKw * sb,
            'P_SG': lambda s, i, t: value(model.p_sg[s, i, t]) * MWtoKw * sb,
            'P_SM': lambda s, i, t: value(model.p_sm[s, i, t]) * MWtoKw * sb,
            'P_BG': lambda s, i, t: value(model.p_bg[s, i, t]) * MWtoKw * sb,
            'P_BM': lambda s, i, t: value(model.p_bm[s, i, t]) * MWtoKw * sb,
        }

        # Variables de Primera Etapa (i, t) -> Results_First_Stage_TSSP.xlsx
        # Binarias (sin escalar)
        vars_1st_binary = {
            'y': lambda i, t: value(model.y[i, t]),
            'w': lambda i, t: value(model.w[i, t]),
        }
        
        # Potencia contratada (escalada) - Indexada por Buses (model.B)
        vars_1st_kappa = {
            'Kappa_BG': lambda i, t: value(model.kappa_bg[i, t]) * MWtoKw * sb,
            'Kappa_SG': lambda i, t: value(model.kappa_sg[i, t]) * MWtoKw * sb,
        }

        # Lista de columnas de tiempo (1, 2, ..., 24) para coincidir con Determinista
        cols_time = [t for t in model.T]

        # ---------------------------------------------------------
        # 2. EXPORTACIÓN POR ESCENARIO (Results_S_x.xlsx)
        # ---------------------------------------------------------
        for s in model.S:
            file_name = os.path.join(base_path, f"Results_S_{s}.xlsx")
            print(f"→ Exportando Escenario {s}: {file_name}")

            with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
                
                # A. Hoja Resumen (Suma diaria por agente)
                resumen_data = []
                for i in model.A:
                    fila = {'Bus': i}
                    for var_name, var_func in vars_per_scenario.items():
                        try:
                            # Sumamos sobre todo T
                            val = sum(var_func(s, i, t) for t in model.T)
                        except:
                            val = 0
                        fila[var_name] = val
                    resumen_data.append(fila)
                
                pd.DataFrame(resumen_data).to_excel(writer, sheet_name='Resumen', index=False)

                # B. Hojas Detalladas (Bus x Hora)
                for var_name, var_func in vars_per_scenario.items():
                    # Creamos estructura vacía
                    data_dict = {}
                    for i in model.A:
                        row_vals = []
                        for t in model.T:
                            try:
                                row_vals.append(round(var_func(s, i, t), 6))
                            except:
                                row_vals.append(0)
                        data_dict[i] = row_vals
                    
                    # Convertir a DataFrame: Indices=Agentes, Columnas=Horas
                    df = pd.DataFrame.from_dict(data_dict, orient='index', columns=cols_time)
                    df.index.name = 'Bus'
                    df.reset_index(inplace=True)
                    df.to_excel(writer, sheet_name=var_name, index=False)

        # ---------------------------------------------------------
        # 3. EXPORTACIÓN PRIMERA ETAPA (Results_First_Stage_TSSP.xlsx)
        # ---------------------------------------------------------
        file_name_1stage = os.path.join(base_path, "Results_First_Stage_TSSP.xlsx")
        print(f"→ Exportando Primera Etapa: {file_name_1stage}")

        with pd.ExcelWriter(file_name_1stage, engine='openpyxl') as writer:
            
            # A. Binarias (Indexadas por Agentes - model.A)
            for var_name, var_func in vars_1st_binary.items():
                data_dict = {}
                for i in model.A:
                    row_vals = []
                    for t in model.T:
                        try: val = round(var_func(i, t), 4) # Binarias no requieren tanta precisión
                        except: val = 0
                        row_vals.append(val)
                    data_dict[i] = row_vals
                
                df = pd.DataFrame.from_dict(data_dict, orient='index', columns=cols_time)
                df.index.name = 'Bus' # Cambiado a 'Bus' para consistencia con análisis
                df.reset_index(inplace=True)
                df.to_excel(writer, sheet_name=var_name, index=False)
            
            # B. Kappa (Indexadas por Buses - model.B)
            for var_name, var_func in vars_1st_kappa.items():
                data_dict = {}
                for i in model.B:
                    row_vals = []
                    for t in model.T:
                        try: val = round(var_func(i, t), 6)
                        except: val = 0
                        row_vals.append(val)
                    data_dict[i] = row_vals

                df = pd.DataFrame.from_dict(data_dict, orient='index', columns=cols_time)
                df.index.name = 'Bus'
                df.reset_index(inplace=True)
                df.to_excel(writer, sheet_name=var_name, index=False)

        print("✅ Exportación completa. Formato alineado con modelo determinista.")

# --- Bloque de ejecución principal (CON CONTADORES) ---
if __name__ == "__main__": 
    
    model_instance = P2P_Model_Stochastic(name="P2P_Stochastic_Run")
    
    try:
        solved_model = model_instance.Solver()
        model_instance.ExportResults(solved_model)
            
    except Exception as e:
        print(f"\n--- ERROR CRÍTICO DURANTE LA EJECUCIÓN ---")
        print(f"Ocurrió un error: {e}")