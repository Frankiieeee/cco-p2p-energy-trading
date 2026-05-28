import numpy as np
import pandas as pd
from pyomo.environ import * 
import pyomo.environ as py
import time
from math import sqrt

'''
Probabilidad Acumulada,Valor Z (aprox.)
50%	0.000
75%	0.674
80%	0.842
85%	1.036
90%	1.282
95%	1.645
99%2.326
'''


class P2P_Model:
    def __init__(self, phi_inv, num_agents=55, name=None):
        self.data = {}
        self.phi_inv = phi_inv
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

    def ReadExcelFile(self, file):
        self.Links = pd.read_excel(file, sheet_name='Adj_Matrix')
        self.Buses = pd.read_excel(file, sheet_name='Buses')
        self.Lines = pd.read_excel(file, sheet_name='Lines')
        self.Scenarios = pd.read_excel(file, sheet_name='Profiles_2')
        self.Circum = pd.read_excel(file, sheet_name='Circum')
        self.A_matrix_wide = pd.read_excel(file, sheet_name='A_matrix_wide')
        self.A_matrix_long = pd.read_excel(file, sheet_name='A_matrix_long')
        self.Data_CCO = pd.read_excel(file, sheet_name='Data_CCO')
   
    ## --------------------------
    
    #crea matrices de nb x nb 
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

##---------------MODEL----------------
    def Model(self):
        model = AbstractModel()


        ## SETS ##
        model.B = py.Set(initialize=np.arange(1,self.buses+1)) 
        model.A = py.Set(initialize=self.Agents)              
        model.refB = py.Set(initialize=np.arange(1,self.refB+1)) 
        model.T = py.Set(initialize=np.arange(1,self.Time+1))
        model.RR = py.Set(initialize=np.arange(1,self.RR+1))   
        
        
        def Set_L_init(model):
            return ((i+1,j+1) for i in np.arange(0,self.buses) for j in np.arange(0,self.buses) if self.Links.loc[i,j]==1 if i<j)
        model.L = py.Set(dimen=2,initialize=Set_L_init)

        def Set_D_init(model,m,n):
            subset = self.A_matrix_long[(self.A_matrix_long['i'] == m) & (self.A_matrix_long['j'] == n) & (self.A_matrix_long['A_ijk'] == 1)]
            return list(subset['k'].astype(int))
        model.D = py.Set(model.L, within=model.B, initialize=Set_D_init)
        
        ## PARAMETERS ##
        model.Phi_inv = py.Param(initialize= self.phi_inv) 

        def A_Matrix_init (model,m,n,i):
            row = self.A_matrix_long[(self.A_matrix_long['i']==m) & (self.A_matrix_long['j']==n) & (self.A_matrix_long['k']==i)]
            return int(row.iloc[0]['A_ijk'])
        model.A_matrix = py.Param(model.L,model.B, rule=A_Matrix_init)

        def PL_init(model,i,t):
            return self.Buses.loc[i-1,'PL']*self.Data_CCO.loc[t-1,'Mu_P_'+str(i)] 
        model.PL_Mu = py.Param(model.A,model.T, rule=PL_init)

        def Var_Error_PL(model,i,t):
            return self.Data_CCO.loc[t-1,'Error_P_'+ str(i)]
        model.Var_error_pl = py.Param(model.A,model.T, rule=Var_Error_PL)

        def Sigma_PL_init(model, i, t):
            return min(sqrt(model.Var_error_pl[i, t])*self.Buses.loc[i-1, 'PL'] / self.sb, model.PL_Mu[i, t]/model.sb)
        model.sigma_pl = py.Param(model.A, model.T, rule=Sigma_PL_init)
        
        def QL_init(model,i,t):   
            if i in model.A:
                return self.Buses.loc[i-1,'QL']*self.Scenarios.loc[t-1,'P_'+str(i)]
            else:
                return 0
        model.QL = py.Param(model.B,model.T, rule=QL_init)

        def QGmin_init(model, i):
            return self.Buses.loc[i-1, 'QGmin'] / model.sb  
        model.QGmin = py.Param(model.B, rule=QGmin_init)

        def QGmax_init(model, i):
            return self.Buses.loc[i-1, 'QGmax'] / model.sb
        model.QGmax = py.Param(model.B, rule=QGmax_init)
        
        def DG_init(model,i):
            return self.Buses.loc[i-1,'DG'] 
        model.DG = py.Param(model.B, rule=DG_init)
        
        def PVmax_init(model,i,t):
            return self.Scenarios.loc[t-1,'PG_max'] 
        model.PV_max = py.Param(model.B,model.T, rule=PVmax_init)
        
        def x_init(model,i,j):
            return self.x_matrix.loc[i-1,j-1]  
        model.X = py.Param(model.B,model.B,rule=x_init)
        
        def r_init(model,i,j):
            return self.r_matrix.loc[i-1,j-1]  
        model.R = py.Param(model.B,model.B,rule=r_init)
        
        def Price_init(mode,t):
            return self.Scenarios.loc[t-1,'Price'] 
        model.Lambda_BG = py.Param(model.T, rule=Price_init)
        
        def Price_SG_init(mode,t):
            return self.Scenarios.loc[t-1,'Price_SG'] 
        model.Lambda_SG = py.Param(model.T, rule=Price_SG_init)
        
        def Smax_init(model,i,j):
            return self.Smax.loc[i-1,j-1]        
        model.s_max = py.Param(model.B,model.B,rule=Smax_init)
        
        def P_SG_max_init(model,i):
            return self.Buses['P_SG_max'].loc[i-1] 
        model.P_SG_max = py.Param(model.B, rule=P_SG_max_init)
        
        def P_BG_max_init(model,i):
            return self.Buses['P_BG_max'].loc[i-1] 
        model.P_BG_max = py.Param(model.B, rule=P_BG_max_init)
        
        def Vmax_init(model,i):
            return self.Buses['Vmax'].loc[i-1]  
        model.Vmax = py.Param(model.B, rule=Vmax_init)
        
        def Vmin_init(model,i):
            return self.Buses['Vmin'].loc[i-1] 
        model.Vmin = py.Param(model.B,  rule=Vmin_init)

        def AA_init(model,i):
            return self.Circum['AA'].loc[i-1]
        model.AA = py.Param(model.RR, rule=AA_init)

        def BB_init(model,i):
            return self.Circum['BB'].loc[i-1]
        model.BB = py.Param(model.RR, rule=BB_init)

        def CC_init(model,i):
            return self.Circum['CC'].loc[i-1]
        model.CC = py.Param(model.RR, rule=CC_init)

        def BT_init(model,i):
            return self.Buses.loc[i-1,'BT']  
        model.BT = py.Param(model.A, rule=BT_init)
        
        def E_BT_init(model,i):
            return self.Buses.loc[i-1,'E_BT']  
        model.E_BT = py.Param(model.A, rule=E_BT_init)

        def PV_Median_init(model,t):
            return self.Data_CCO['PV_Median'].loc[t-1]
        model.PV_Median = py.Param(model.T, rule=PV_Median_init)

        def Var_Error_PV(model,t):
            return self.Data_CCO['Varianza_Error_PV'].loc[t-1]
        model.Var_error_pv = py.Param(model.T, rule=Var_Error_PV)

        def Sigma_PV_init(model, i, t):
            return min(sqrt(model.Var_error_pv[t])*model.DG[i]/100, model.PV_max[i,t]*model.DG[i]/100)
        model.sigma_pv = py.Param(model.A, model.T, rule=Sigma_PV_init)

        model.rho_pv = py.Param(model.T, initialize=lambda m, t: 1.0)
        model.rho_pl = py.Param(model.T, initialize=lambda m, t: 1.0)
        model.rho_pv_pl = py.Param(model.T, initialize=lambda m, t: 0.0)

        model.Fi = py.Param(initialize = self.Fi) 
        model.SOC_min = self.SOC_min   
        model.SOC_max = self.SOC_max
        model.PB = self.PB        

        def Var_PV_sum_t(model, t):
            sigma_sq_sum = sum(model.sigma_pv[i, t]**2 for i in model.A)
            sigma_sum = sum(model.sigma_pv[i, t] for i in model.A)
            rho = model.rho_pv[t]
            return (1 - rho) * sigma_sq_sum + rho * (sigma_sum ** 2)
        model.var_pv_sum = py.Param(model.T, initialize=Var_PV_sum_t)
        
        def Sigma_DeltaP_agent(model, i, t):
            return sqrt( max(0.0, model.sigma_pv[i, t]**2 + model.sigma_pl[i, t]**2 - 2 * model.rho_pv_pl[t] * model.sigma_pv[i, t] * model.sigma_pl[i, t]) )
        model.sigma_dp_agent = py.Param(model.A, model.T, rule=Sigma_DeltaP_agent)

        def Sigma_p_line_init(model, i, j, t):
            downstream_agents = [k for k in model.D[i, j] if k in model.A]
            if not downstream_agents:
                return 0.0
            return sqrt(sum(model.sigma_dp_agent[k, t]**2 for k in downstream_agents))
        model.sigma_p = py.Param(model.L, model.T, rule=Sigma_p_line_init)

        def Sigma_q_line_init(model, i, j, t):
            return 0.0
        model.sigma_q = py.Param(model.L, model.T, rule=Sigma_q_line_init)

        def Cov_pq_line_init(model, i, j, t):
            return 0.0
        model.cov_pq = py.Param(model.L, model.T, rule=Cov_pq_line_init)

        def Sigma_v_init(model, i, t):
            sigma_total = 0.0
            for (m, n) in model.L:
                if model.A_matrix[m, n, i] == 1:
                    downstream_agents = [k for k in model.D[m, n] if k in model.A]
                    if downstream_agents:
                        sigma_line = sqrt(sum(model.sigma_dp_agent[k, t]**2 for k in downstream_agents))
                        sigma_total += sqrt((2*model.R[m, n]*sigma_line)**2) 
            return sigma_total
        model.sigma_v = py.Param(model.B, model.T, rule=Sigma_v_init)

        def Sigma_qg_init(model, i, t):
            return 0.0
        model.sigma_qg = py.Param(model.B, model.T, rule=Sigma_qg_init)

        def SigmaSum_PV_t(model, t):
            return sum(model.sigma_pv[i, t] for i in model.A)
        model.sigma_pv_sum = py.Param(model.T, initialize=SigmaSum_PV_t)

        def SigmaSum_PL_t(model, t):
            return sum(model.sigma_pl[i, t] for i in model.A)
        model.sigma_pl_sum = py.Param(model.T, initialize=SigmaSum_PL_t)

        def Var_PL_sum_t(model, t):
            sig2_sum = sum(model.sigma_pl[i, t]**2 for i in model.A)
            sig_sum  = sum(model.sigma_pl[i, t]     for i in model.A)
            rho = model.rho_pl[t]
            return (1 - rho) * sig2_sum + rho * (sig_sum ** 2)
        model.var_pl_sum = py.Param(model.T, initialize=Var_PL_sum_t)

        def Sigma_SOC_init(model, i, t):
            if model.BT[i] == 0:
                return 0.0
            sigma_soc_val = sqrt((model.Fi)**2 * model.sigma_pv[i, t]**2 + (1 / model.Fi)**2 * model.sigma_pl[i, t]**2) 
            return sigma_soc_val
        model.sigma_soc = py.Param(model.A, model.T, rule=Sigma_SOC_init)
 

        ##---------------------------------------------------------------------
        model.sb = 100  
        model.MWtoKW = 1000   
        model.nA = self.nA
        
        ## VARIABLES ##
        model.pv = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0) 
        model.qg = py.Var(model.B,model.T, within = py.Reals, initialize = 0)
        model.v = py.Var(model.B,model.T, within = py.NonNegativeReals, initialize = 0)
        model.p = py.Var(model.L,model.T) 
        model.q = py.Var(model.L,model.T) 
        model.kappa_bg = py.Var(model.B,model.T, within = py.NonNegativeReals, initialize = 0) 
        model.kappa_sg = py.Var(model.B,model.T, within = py.NonNegativeReals, initialize = 0)

        model.dp =  py.Var(model.A,model.T, within = py.Reals, initialize = 0) 
        model.dp_pos = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0) 
        model.dp_neg = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0) 

        model.y = py.Var(model.A,model.T, within = py.Binary, initialize = 0) 
        model.pl = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0) 

        model.soc = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0)   
        model.ch = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0)    
        model.ds = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0)   
        model.w = py.Var(model.A,model.T, within = py.Binary, initialize = 0)  

        model.p_sg = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0)
        model.p_sm = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0)
        model.p_bg = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0)
        model.p_bm = py.Var(model.A,model.T, within = py.NonNegativeReals, initialize = 0)

        ## OBJECTIVE FUNCTION ##
        def Fun_obj(model):
            return  sum(model.Lambda_BG[t]*(model.kappa_bg[i,t]*model.sb) - model.Lambda_SG[t]*(model.kappa_sg[i,t]*model.sb) for i in model.B for t in model.T) + \
                sum(model.pv[i,t]*model.sb*0.01  for i in model.A for t in model.T) 
        model.FunObj = py.Objective(rule = Fun_obj, sense = py.minimize)
    ##restriciones
    
        def NPFE_R (model,i,t):
            value1 = sum(model.p[m,n,t] for (m,n) in model.L if m==i) - sum(model.p[n,m,t]  for (n,m) in model.L if m==i)
            if i in model.A: 
                return model.dp[i,t]  == value1 
            else:
                return model.kappa_bg[i,t] - model.kappa_sg[i,t] == value1
        model.npfe_r = py.Constraint(model.B,model.T, rule=NPFE_R) 

        def NPFE_Im (model,i,t):
            value2 = sum(model.q[m,n,t] for (m,n) in model.L if m==i) - sum(model.q[n,m,t]  for (n,m) in model.L if m==i) 
            return model.qg[i,t] - model.QL[i,t]/model.sb == value2
        model.npfe_im = py.Constraint(model.B,model.T, rule=NPFE_Im) 

        def LPFE_R (model, i,j,t):                                                
            return model.v[j,t] == model.v[i,t] -2*(model.R[i,j]*model.p[i,j,t]+model.X[i,j]*model.q[i,j,t])
        model.lpfe_r = py.Constraint(model.L,model.T, rule=LPFE_R)

        def DP_init (model,i,t):
            return model.dp[i,t] == model.pv[i, t] - model.pl[i,t] + model.ds[i,t] - model.ch[i,t]  
        model.dp_init = py.Constraint(model.A, model.T, rule=DP_init)

        def DP_init2 (model,i,t):
            return model.dp[i,t] == model.dp_pos[i,t] - model.dp_neg[i,t] 
        model.dp_init2 = py.Constraint(model.A, model.T, rule=DP_init2)

        def bus_uno_compra (model,i,t):
            return model.kappa_bg[i,t] <= model.P_BG_max[i]
        model.bus_uno_buy=Constraint(model.B,model.T,rule=bus_uno_compra)
        
        def bus_uno_venta (model,i,t):
            return model.kappa_sg[i,t] <= model.P_SG_max[i]
        model.bus_uno_sell=Constraint(model.B,model.T,rule=bus_uno_venta)

        def DP_Pos(model, i, t):
            return model.dp_pos[i, t]   <=  ((model.PV_max[i, t]) * model.DG[i]*200) / model.sb * model.y[i,t] 
        model.DP_Pos = py.Constraint(model.A, model.T, rule=DP_Pos)

        def DP_Neg(model, i, t):
            return model.dp_neg[i, t]  <=  (model.PL_Mu[i, t]*200/model.sb) * (1 - model.y[i,t]) 
        model.DP_Neg = py.Constraint(model.A, model.T, rule=DP_Neg)

        def SOCR (model,i,t): 
            if t == 1:
                return model.soc[i,t] == 0.5 * (model.BT[i]/model.sb) + (model.Fi*model.ch[i,t]) - (model.ds[i,t]/model.Fi)
            else:
                return model.soc[i,t] == model.soc[i,t-1] + model.Fi*model.ch[i,t] - model.ds[i,t]/model.Fi
        model.socr = Constraint(model.A, model.T, rule=SOCR)

        def CHR (model,i,t): 
            return model.ch[i,t] <= (model.PB/model.sb) * model.w[i,t] 
        model.chr = Constraint(model.A,model.T, rule=CHR)

        def DSR (model,i,t): 
            return model.ds[i,t] <= (model.PB/model.sb)*(1-model.w[i,t])
        model.dsr = Constraint(model.A,model.T, rule=DSR)

        def DSR_W (model,i,t): 
            return model.w[i,t] <= model.E_BT[i]
        model.dsr_w = Constraint(model.A,model.T, rule=DSR_W)

        def DP_Pos_2(model, i, t):
            return model.dp_pos[i, t]   == model.p_sg[i, t]  + model.p_sm[i, t] 
        model.DP_Pos_2 = py.Constraint(model.A, model.T, rule=DP_Pos_2)

        def DP_Neg_2(model, i, t):
            return model.dp_neg[i, t]  == model.p_bg[i, t]  + model.p_bm[i, t] 
        model.DP_Neg_2 = py.Constraint(model.A, model.T, rule=DP_Neg_2)

        def Market_Balance(model, t):
            return sum(model.p_sm[i, t] for i in model.A) == sum(model.p_bm[i, t] for i in model.A) 
        model.market_balance = py.Constraint(model.T, rule=Market_Balance)

        def Kappa_BG_Balance(model, t):
            return sum(model.p_bg[i, t] for i in model.A) == sum(model.kappa_bg[i, t] for i in model.B) 
        model.kappa_bg_balance = py.Constraint(model.T, rule=Kappa_BG_Balance)

        def Kappa_SG_Balance(model, t):
            return sum(model.p_sg[i, t] for i in model.A) == sum(model.kappa_sg[i, t] for i in model.B) 
        model.kappa_sg_balance = py.Constraint(model.T, rule=Kappa_SG_Balance)

        ###-----------------------CCO---------------------------------------

        def QG_max_CCO(model, i, t):
            return model.qg[i, t] + model.Phi_inv * model.sigma_qg[i, t] <= model.QGmax[i]
        model.QG_max_CCO = py.Constraint(model.B, model.T, rule=QG_max_CCO)

        def QG_min_CCO(model, i, t):
            return model.qg[i, t] - model.Phi_inv * model.sigma_qg[i, t] >= model.QGmin[i]
        model.QG_min_CCO = py.Constraint(model.B, model.T, rule=QG_min_CCO)

        def V_max_CCO(model, i, t):
            return model.v[i, t] + model.Phi_inv * model.sigma_v[i, t] <= model.Vmax[i]**2
        model.V_max_CCO = py.Constraint(model.B, model.T, rule=V_max_CCO)

        def V_min_CCO(model, i, t):
            return model.v[i, t] - model.Phi_inv * model.sigma_v[i, t] >= model.Vmin[i]**2
        model.V_min_CCO = py.Constraint(model.B, model.T, rule=V_min_CCO)

        def PV_gen_CCO(model, i, t):
            return model.pv[i, t] + model.Phi_inv * model.sigma_pv[i, t] * model.DG[i] / model.sb <= (model.PV_max[i, t]) * model.DG[i] / model.sb
        model.PV_gen_CCO = py.Constraint(model.A, model.T, rule=PV_gen_CCO)

        def PL_CCO_1(model, i, t):
            return model.pl[i, t] - model.Phi_inv * model.sigma_pl[i, t] >= model.PL_Mu[i, t]/model.sb
        model.pl_cco_1 = py.Constraint(model.A, model.T, rule=PL_CCO_1)

        def Smax_CCO(model, i, j, t, r):
            Omega = sqrt( (model.AA[r]**2) * (model.sigma_p[i, j, t]**2) + (model.BB[r]**2) * (model.sigma_q[i, j, t]**2) + 2 * model.AA[r] * model.BB[r] * model.cov_pq[i, j, t] )
            return model.AA[r] * model.p[i, j, t] + model.BB[r] * model.q[i, j, t] + model.CC[r] * (model.s_max[i, j] / model.sb) + model.Phi_inv * Omega <= 0
        model.Smax_CCO = py.Constraint(model.L, model.T, model.RR, rule=Smax_CCO)

        def Limit_SOC_CCO (model,i,t): 
            return model.soc[i,t] - model.Phi_inv * model.sigma_soc[i, t] >= (model.BT[i]/model.sb)*model.SOC_min
        model.limit_soc_cco = Constraint(model.A,model.T, rule=Limit_SOC_CCO)

        def Limit_SOCsup_CCO (model,i, t ):
            return model.soc[i,t] + model.Phi_inv * model.sigma_soc[i, t] <= (model.BT[i]/model.sb)*model.SOC_max
        model.limit_socsup_cco = Constraint(model.A,model.T, rule=Limit_SOCsup_CCO)
      
        return model.create_instance()


    def get_summary_dict(self, model, error_val, prob_val):
        from pyomo.environ import value
        
        # Cálculos de costos base
        costo_compra = sum(value(model.kappa_bg[i,t]) * model.Lambda_BG[t] * self.sb for i in model.B for t in model.T)
        costo_venta = sum(value(model.kappa_sg[i,t]) * model.Lambda_SG[t] * self.sb for i in model.B for t in model.T)
        costo_bateria = sum((value(model.pv[i,t]) + value(model.ds[i,t])) * 0.01 * self.sb for i in model.A for t in model.T)
        
        # Cálculos operativos
        voltajes = [value(model.v[i,t]) for i in model.B for t in model.T]
        margen_pv = sum((model.PV_max[i,t] * model.DG[i] / self.sb - value(model.pv[i,t])) * self.sb for i in model.A for t in model.T)
        
        return {
            'Escenario_Error_%': error_val,
            'Nivel_Confianza_%': prob_val,
            'Valor_Z_Inversa': self.phi_inv,
            'F_Objetivo_Total_[CLP]': round(value(model.FunObj), 2),
            'Costo_Compra_Red_[CLP]': round(costo_compra, 2),
            'Ingreso_Venta_Red_[CLP]': round(costo_venta, 2),
            'Costo_Operativo_[CLP]': round(costo_bateria, 2),
            'E_Comprada_[kWh]': round(sum(value(model.kappa_bg[i,t]) * self.sb for i in model.B for t in model.T), 2),
            'E_Vendida_[kWh]': round(sum(value(model.kappa_sg[i,t]) * self.sb for i in model.B for t in model.T), 2),
            'Gen_PV_Total_[kWh]': round(sum(value(model.pv[i,t]) * self.sb for i in model.A for t in model.T), 2),
            'Demanda_Satisfecha_[kWh]': round(sum(value(model.pl[i,t]) * self.sb for i in model.A for t in model.T), 2),
            'Intercambio_P2P_[kWh]': round(sum(value(model.p_sm[i,t]) * self.sb for i in model.A for t in model.T), 2),
            'Voltaje_Promedio_[p.u.]': round(sum(voltajes) / len(voltajes), 4),
            'Voltaje_Minimo_[p.u.]': round(min(voltajes), 4),
            'Margen_Reserva_PV_[kW]': round(margen_pv, 2)
        }



    def Solver(self):
        Model = self.Model()
        self.opt = SolverFactory('gurobi')
        start_solve_time = time.time() 
        
        results_Model = self.opt.solve(Model, tee=True)
        results_Model.write()
        solve_duration = time.time() - start_solve_time      
        print(f"\n--- Optimización Finalizada. Tiempo de Gurobi: {solve_duration:.2f} segundos ---")
        print('Objective Function Result: ' + str(round(value(Model.FunObj), 8)))
        return Model


    def ExportResults(self, model, output_name='Results_PVandPL_BESS_P2P.xlsx'):
        import pandas as pd
        from pyomo.environ import value

        Buses_Table = pd.DataFrame()
        Lineas_Table = pd.DataFrame()

        Agentes_Table_PV = pd.DataFrame()
        Agentes_Table_PL = pd.DataFrame()
        Agentes_Table_pl = pd.DataFrame()
        Agentes_Table_DP = pd.DataFrame()
        Agentes_Table_DP_pos = pd.DataFrame()
        Agentes_Table_DP_neg = pd.DataFrame()
        Table_Sigma_PV = pd.DataFrame()
        Table_PV_Mu = pd.DataFrame()

        Table_Kappa_bg = pd.DataFrame()
        Table_Kappa_sg = pd.DataFrame()

        Table_Var_Error_pl = pd.DataFrame()
        Table_Sigma_pl = pd.DataFrame()
        Table_Sigma_dp_agente = pd.DataFrame()

        Table_DS = pd.DataFrame()
        Table_CH = pd.DataFrame()

        Tabla_Sigma_soc = pd.DataFrame()
        Tabla_soc = pd.DataFrame()

        Agentes_Table_PSG = pd.DataFrame()
        Agentes_Table_PSM = pd.DataFrame()
        Agentes_Table_PBG = pd.DataFrame()
        Agentes_Table_PBM = pd.DataFrame()

        r = 0
        for a in model.A:
            for t in model.T:
                Agentes_Table_PV.loc[r,'Bus'] = a
                Agentes_Table_PV.loc[r,t] = round(value(model.pv[a, t])*self.MWtoKw*self.sb, 8)

                Agentes_Table_PL.loc[r,'Bus'] = a
                Agentes_Table_PL.loc[r,t] = round(model.PL_Mu[a, t]*self.MWtoKw, 8)

                Agentes_Table_pl.loc[r,'Bus'] = a
                Agentes_Table_pl.loc[r,t] = round(value(model.pl[a, t])*self.MWtoKw*self.sb, 8)

                Agentes_Table_DP.loc[r,'Bus'] = a
                Agentes_Table_DP.loc[r,t] = round(value(model.dp[a, t])*self.MWtoKw*self.sb, 8)

                Agentes_Table_DP_pos.loc[r,'Bus'] = a
                Agentes_Table_DP_pos.loc[r,t] = round(value(model.dp_pos[a, t])*self.MWtoKw*self.sb, 8)

                Agentes_Table_DP_neg.loc[r,'Bus'] = a
                Agentes_Table_DP_neg.loc[r,t] = round(value(model.dp_neg[a, t])*self.MWtoKw*self.sb, 8)

                Table_Sigma_PV.loc[r,'Bus'] = a
                Table_Sigma_PV.loc[r,t] = round(model.sigma_pv[a, t]*self.MWtoKw*self.sb, 8)

                Table_Var_Error_pl.loc[r,'Bus'] = a
                Table_Var_Error_pl.loc[r,t] = round(model.Var_error_pl[a, t]*self.MWtoKw*self.sb, 8)

                Table_Sigma_pl.loc[r,'Bus'] = a
                Table_Sigma_pl.loc[r,t] = round(model.sigma_pl[a, t]*self.MWtoKw*self.sb, 8)

                Table_Sigma_dp_agente.loc[r,'Bus'] = a
                Table_Sigma_dp_agente.loc[r,t] = round(model.sigma_dp_agent[a, t]*self.MWtoKw*self.sb, 8)

                Table_PV_Mu.loc[r,'Bus'] = a
                Table_PV_Mu.loc[r,t] = round((model.PV_max[a, t]) * model.DG[a]*self.MWtoKw, 8)

                Table_DS.loc[r,'Bus'] = a
                Table_DS.loc[r,t] = round(value(model.ds[a, t])*self.MWtoKw*self.sb, 8)

                Table_CH.loc[r,'Bus'] = a
                Table_CH.loc[r,t] = round(value(model.ch[a, t])*self.MWtoKw*self.sb, 8)

                Tabla_Sigma_soc.loc[r,'Bus'] = a
                Tabla_Sigma_soc.loc[r,t] = round(value(model.sigma_soc[a, t])*self.MWtoKw*self.sb, 8)

                Tabla_soc.loc[r,'Bus'] = a
                Tabla_soc.loc[r,t] = round(value(model.soc[a, t])*self.MWtoKw*self.sb, 8)
                
                Agentes_Table_PSG.loc[r,'Bus'] = a
                Agentes_Table_PSG.loc[r,t] = round(value(model.p_sg[a, t])*self.MWtoKw*self.sb, 8)

                Agentes_Table_PSM.loc[r,'Bus'] = a
                Agentes_Table_PSM.loc[r,t] = round(value(model.p_sm[a, t])*self.MWtoKw*self.sb, 8)

                Agentes_Table_PBG.loc[r,'Bus'] = a
                Agentes_Table_PBG.loc[r,t] = round(value(model.p_bg[a, t])*self.MWtoKw*self.sb, 8)

                Agentes_Table_PBM.loc[r,'Bus'] = a
                Agentes_Table_PBM.loc[r,t] = round(value(model.p_bm[a, t])*self.MWtoKw*self.sb, 8)

            r += 1

        r = 0
        for b in model.B:
            for t in model.T:
                Table_Kappa_bg.loc[r,'Bus'] = b
                Table_Kappa_bg.loc[r,t] = round(value(model.kappa_bg[b, t])*self.MWtoKw*self.sb, 8)

                Table_Kappa_sg.loc[r,'Bus'] = b
                Table_Kappa_sg.loc[r,t] = round(value(model.kappa_sg[b, t])*self.MWtoKw*self.sb, 8)

            r += 1

        r = 0
        for b in model.B:
            for t in model.T:
                Buses_Table.loc[r, 'Bus'] = b
                Buses_Table.loc[r, 'Tiempo'] = t
                Buses_Table.loc[r, 'v'] = round(value(model.v[b, t]), 8)
                Buses_Table.loc[r, 'q_bg'] = round(value(model.qg[b, t])*self.MWtoKw*self.sb, 8)
                r += 1
   
        r = 0
        for l in model.L:
            for t in model.T:
                Lineas_Table.loc[r, 'Linea'] = f"{l[0]}-{l[1]}"
                Lineas_Table.loc[r, 'Tiempo'] = t
                Lineas_Table.loc[r, 'p'] = round(value(model.p[l, t])*self.MWtoKw*self.sb, 8)
                Lineas_Table.loc[r, 'q'] = round(value(model.q[l, t])*self.MWtoKw*self.sb, 8)
                r += 1

        with pd.ExcelWriter(output_name) as writer:
            Buses_Table.to_excel(writer, sheet_name='Buses', index=False)
            Lineas_Table.to_excel(writer, sheet_name='Lineas', index=False)
            Agentes_Table_PV.to_excel(writer, sheet_name='PV', index=False)
            Agentes_Table_PL.to_excel(writer, sheet_name='PL', index=False)
            Agentes_Table_pl.to_excel(writer, sheet_name='pl', index=False)
            Agentes_Table_DP.to_excel(writer, sheet_name='DP', index=False)
            Agentes_Table_DP_pos.to_excel(writer, sheet_name='DP_pos', index=False)
            Agentes_Table_DP_neg.to_excel(writer, sheet_name='DP_neg', index=False)
            Table_Sigma_PV.to_excel(writer, sheet_name='Sigma_PV', index=False)
            Table_Kappa_bg.to_excel(writer, sheet_name='Kappa_BG', index=False)
            Table_Kappa_sg.to_excel(writer, sheet_name='Kappa_SG', index=False)

            Tabla_Sigma_soc.to_excel(writer, sheet_name='Sigma_soc', index=False)
            Tabla_soc.to_excel(writer, sheet_name='soc', index=False)

            Table_DS.to_excel(writer, sheet_name='DS', index=False)
            Table_CH.to_excel(writer, sheet_name='CH', index=False)

            Table_PV_Mu.to_excel(writer, sheet_name='PV_Mu', index=False)

            Table_Var_Error_pl.to_excel(writer, sheet_name='Var_Eror_pl', index=False)
            Table_Sigma_pl.to_excel(writer, sheet_name='Sigma_pl', index=False)
            Table_Sigma_dp_agente.to_excel(writer, sheet_name='Sigma_dp_agente', index=False)

            Agentes_Table_PSG.to_excel(writer, sheet_name='P_SG', index=False)
            Agentes_Table_PSM.to_excel(writer, sheet_name='P_SM', index=False)
            Agentes_Table_PBG.to_excel(writer, sheet_name='P_BG', index=False)
            Agentes_Table_PBM.to_excel(writer, sheet_name='P_BM', index=False)


if __name__ == '__main__':
    import os
    import pandas as pd

    # ---------------------------------------------------------------------------
    # Path configuration — all paths are relative to the repository root.
    # Run this script from the repo root:  python src/CCO_Model_PVandPL_BESS_P2P.py
    # ---------------------------------------------------------------------------
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
    RESULTS_DIR  = os.path.join(ROOT_DIR, 'data', 'processed', 'sensitivity')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Forecast error levels → input file mapping
    archivos_error = {
        '05': 'Case_Study_05.xlsx',
        '10': 'Case_Study_10.xlsx',
        '15': 'Case_Study_15.xlsx',
    }

    # Confidence levels → Φ⁻¹(1-η) quantile values
    niveles_confianza = {50: 0.000, 75: 0.674, 90: 1.282, 95: 1.645, 99: 2.326}

    resultados_globales = []

    for error_key, excel_name in archivos_error.items():
        for prob_key, z_val in niveles_confianza.items():
            print(f"\n{'='*55}")
            print(f"  Model: Error {error_key}% | Confidence {prob_key}% | Z = {z_val}")
            print(f"{'='*55}")

            # 1. Instantiate with the corresponding quantile value
            running = P2P_Model(phi_inv=z_val)

            # 2. Load data
            ruta_excel = os.path.join(DATA_RAW_DIR, excel_name)
            running.ReadExcelFile(ruta_excel)

            # 3. Solve with Gurobi
            modelo = running.Solver()

            # 4. Export per-run detail results
            nombre_salida = os.path.join(
                RESULTS_DIR, f'Results_Error{error_key}_Conf{prob_key}.xlsx'
            )
            running.ExportResults(modelo, output_name=nombre_salida)

            # 5. Collect summary metrics
            resumen_corrida = running.get_summary_dict(modelo, error_key, prob_key)
            resultados_globales.append(resumen_corrida)

    # 6. Write consolidated sensitivity report
    df_consolidado = pd.DataFrame(resultados_globales)
    ruta_consolidado = os.path.join(ROOT_DIR, 'data', 'processed', 'Resumen_Global_Sensibilidad_CCO.xlsx')
    df_consolidado.to_excel(ruta_consolidado, index=False)

    print("\nAll simulations completed successfully.")
    print(f"Consolidated report saved to: {ruta_consolidado}")