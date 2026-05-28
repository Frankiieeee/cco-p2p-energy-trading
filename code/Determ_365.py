import numpy as np
import pandas as pd
from pyomo.environ import *
import pyomo.environ as py
import os

class P2P_Model_Deterministic:
    def __init__(self, name=None):
        self.data = {}
        self.Time = 24
        self.refB = 1
        self.RR = 12
        self.MWtoKw = 1000
        self.sb = 100
        self.Agents = [7,8,10,11,18,19,22,25,26,28,31,32,38,41,42,45,46,51,53,57,58,60,62,
                       74,83,84,94,95,109,110,127,130,133,137,138,142,144,146,148,151,153,
                       154,157,158,160,165,166,169,170,173,174,177,178,191,192]
        self.nA = len(self.Agents)
        self.BigM = 1000
        self.SOC_min = 0.1
        self.SOC_max = 0.9
        self.PB = 0.0034
        self.Fi_val = 0.9

    def ReadExcelFile(self, file):
        print("Reading Excel file...")
        self.Links   = pd.read_excel(file, sheet_name='Adj_Matrix', index_col=0)
        self.Buses   = pd.read_excel(file, sheet_name='Buses')
        self.Lines   = pd.read_excel(file, sheet_name='Lines')
        self.Circum  = pd.read_excel(file, sheet_name='Circum')

        print("Loading annual profiles...")
        self.Profiles_Raw = pd.read_excel(file, sheet_name='Profiles')
        self.PV_Raw       = pd.read_excel(file, sheet_name='PV')
        self.Price_Raw    = pd.read_excel(file, sheet_name='Price')

        col_id = self.Profiles_Raw.columns[0]
        self.Profiles_Raw['Agent'] = self.Profiles_Raw[col_id].apply(lambda x: int(str(x).split('_')[1]))
        self.Profiles_Raw['Day']   = self.Profiles_Raw[col_id].apply(lambda x: int(str(x).split('_')[2]))
        self.Profiles_Raw.set_index(['Day', 'Agent'], inplace=True)

        self.buses = len(self.Buses)
        self.x_matrix = pd.DataFrame(np.zeros([self.buses, self.buses]))
        self.r_matrix = pd.DataFrame(np.zeros([self.buses, self.buses]))
        self.Smax     = pd.DataFrame(np.zeros([self.buses, self.buses]))

        for i in range(len(self.Lines)):
            fb = self.Lines.loc[i, 'fbus']
            tb = self.Lines.loc[i, 'tbus']
            self.x_matrix.loc[fb-1, tb-1] = self.x_matrix.loc[tb-1, fb-1] = self.Lines.loc[i, 'x']
            self.r_matrix.loc[fb-1, tb-1] = self.r_matrix.loc[tb-1, fb-1] = self.Lines.loc[i, 'r']
            self.Smax.loc[fb-1, tb-1]     = self.Smax.loc[tb-1, fb-1]     = self.Lines.loc[i, 'Smax']

        print("Data loaded.")

    def Run365(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.opt = SolverFactory('gurobi')
        self.opt.options['FeasibilityTol'] = 1e-9

        for day in range(1, 366):
            print(f"\n--- Day {day}/365 ---")
            try:
                self.Data_CCO = self.build_daily_data(day)
                model = self.Model()
                self.opt.solve(model)
                print(f"Day {day} solved. Obj: {round(value(model.FunObj), 4)}")
                out_file = os.path.join(output_dir, f"Results_Day_{day:03d}.xlsx")
                self.ExportResults(model, out_file)
            except Exception as e:
                print(f"ERROR on day {day}: {e}")

    def build_daily_data(self, day):
        df_day = self.Price_Raw[['Price', 'Price_SG']].copy()

        if 'day_of_year' in self.PV_Raw.columns:
            pv_row = self.PV_Raw[self.PV_Raw['day_of_year'] == day]
        else:
            pv_row = self.PV_Raw.iloc[[day - 1]]

        pv_cols = [c for c in self.PV_Raw.columns if str(c).isdigit() and int(c) <= 24]
        df_day['PG_max'] = pv_row[pv_cols].values.flatten()

        try:
            day_profiles = self.Profiles_Raw.loc[day]
        except KeyError:
            raise ValueError(f"No profiles found for day {day}")

        hourly_cols = [c for c in day_profiles.columns if str(c).isdigit() and int(c) <= 24]
        pl_data = day_profiles[hourly_cols].T
        pl_data.columns = [f'P_{agent}' for agent in pl_data.columns]

        df_day.reset_index(drop=True, inplace=True)
        pl_data.reset_index(drop=True, inplace=True)
        return pd.concat([df_day, pl_data], axis=1)

    def Model(self):
        model = AbstractModel()

        model.B    = py.Set(initialize=np.arange(1, self.buses + 1))
        model.A    = py.Set(initialize=self.Agents)
        model.refB = py.Set(initialize=np.arange(1, self.refB + 1))
        model.T    = py.Set(initialize=np.arange(1, self.Time + 1))
        model.RR   = py.Set(initialize=np.arange(1, self.RR + 1))

        def Set_L_init(model):
            return ((i+1, j+1) for i in np.arange(0, self.buses) for j in np.arange(0, self.buses)
                    if self.Links.iloc[i, j] == 1 and i < j)
        model.L = py.Set(dimen=2, initialize=Set_L_init)

        def PL_init(model, i, t):
            return self.Buses.loc[i-1, 'PL'] * self.Data_CCO.loc[t-1, 'P_' + str(i)]
        model.PL_Mu = py.Param(model.A, model.T, rule=PL_init)

        def QL_init(model, i, t):
            if i in model.A:
                return self.Buses.loc[i-1, 'QL'] * self.Data_CCO.loc[t-1, 'P_' + str(i)]
            return 0
        model.QL = py.Param(model.B, model.T, rule=QL_init)

        model.QGmin  = py.Param(model.B, rule=lambda m, i: self.Buses.loc[i-1, 'QGmin'] / m.sb)
        model.QGmax  = py.Param(model.B, rule=lambda m, i: self.Buses.loc[i-1, 'QGmax'] / m.sb)
        model.DG     = py.Param(model.B, rule=lambda m, i: self.Buses.loc[i-1, 'DG'])
        model.PV_max = py.Param(model.B, model.T, rule=lambda m, i, t: self.Data_CCO.loc[t-1, 'PG_max'])
        model.X      = py.Param(model.B, model.B, rule=lambda m, i, j: self.x_matrix.loc[i-1, j-1])
        model.R      = py.Param(model.B, model.B, rule=lambda m, i, j: self.r_matrix.loc[i-1, j-1])
        model.Lambda_BG  = py.Param(model.T, rule=lambda m, t: self.Data_CCO.loc[t-1, 'Price'])
        model.Lambda_SG  = py.Param(model.T, rule=lambda m, t: self.Data_CCO.loc[t-1, 'Price_SG'])
        model.s_max      = py.Param(model.B, model.B, rule=lambda m, i, j: self.Smax.loc[i-1, j-1])
        model.P_SG_max   = py.Param(model.B, rule=lambda m, i: self.Buses['P_SG_max'].loc[i-1])
        model.P_BG_max   = py.Param(model.B, rule=lambda m, i: self.Buses['P_BG_max'].loc[i-1])
        model.Vmax       = py.Param(model.B, rule=lambda m, i: self.Buses['Vmax'].loc[i-1])
        model.Vmin       = py.Param(model.B, rule=lambda m, i: self.Buses['Vmin'].loc[i-1])
        model.AA = py.Param(model.RR, rule=lambda m, i: self.Circum['AA'].loc[i-1])
        model.BB = py.Param(model.RR, rule=lambda m, i: self.Circum['BB'].loc[i-1])
        model.CC = py.Param(model.RR, rule=lambda m, i: self.Circum['CC'].loc[i-1])
        model.BT    = py.Param(model.A, rule=lambda m, i: self.Buses.loc[i-1, 'BT'])
        model.E_BT  = py.Param(model.A, rule=lambda m, i: self.Buses.loc[i-1, 'E_BT'])
        model.Fi    = py.Param(initialize=self.Fi_val)
        model.SOC_min = self.SOC_min
        model.SOC_max = self.SOC_max
        model.PB  = self.PB
        model.M   = self.BigM
        model.sb  = 100
        model.MWtoKW = 1000
        model.nA  = self.nA

        model.pv       = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.qg       = py.Var(model.B, model.T, within=py.Reals, initialize=0)
        model.v        = py.Var(model.B, model.T, within=py.NonNegativeReals, initialize=0)
        model.p        = py.Var(model.L, model.T)
        model.q        = py.Var(model.L, model.T)
        model.kappa_bg = py.Var(model.B, model.T, within=py.NonNegativeReals, initialize=0)
        model.kappa_sg = py.Var(model.B, model.T, within=py.NonNegativeReals, initialize=0)
        model.dp       = py.Var(model.A, model.T, within=py.Reals, initialize=0)
        model.dp_pos   = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.dp_neg   = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.y        = py.Var(model.A, model.T, within=py.Binary, initialize=0)
        model.pl       = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.soc      = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.ch       = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.ds       = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.w        = py.Var(model.A, model.T, within=py.Binary, initialize=0)
        model.p_sg     = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.p_sm     = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.p_bg     = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)
        model.p_bm     = py.Var(model.A, model.T, within=py.NonNegativeReals, initialize=0)

        def Fun_obj(model):
            return (sum(model.Lambda_BG[t] * model.kappa_bg[i,t] * model.sb
                        - model.Lambda_SG[t] * model.kappa_sg[i,t] * model.sb
                        for i in model.B for t in model.T)
                    + sum((model.ds[i,t] + model.pv[i,t]) * model.sb * 0.01
                          for i in model.A for t in model.T))
        model.FunObj = py.Objective(rule=Fun_obj, sense=py.minimize)

        def NPFE_R(model, i, t):
            val = (sum(model.p[m,n,t] for (m,n) in model.L if m == i)
                   - sum(model.p[n,m,t] for (n,m) in model.L if m == i))
            if i in model.A:
                return model.dp[i,t] == val
            return model.kappa_bg[i,t] - model.kappa_sg[i,t] == val
        model.npfe_r = py.Constraint(model.B, model.T, rule=NPFE_R)

        def NPFE_Im(model, i, t):
            val = (sum(model.q[m,n,t] for (m,n) in model.L if m == i)
                   - sum(model.q[n,m,t] for (n,m) in model.L if m == i))
            return model.qg[i,t] - model.QL[i,t] / model.sb == val
        model.npfe_im = py.Constraint(model.B, model.T, rule=NPFE_Im)

        def LPFE_R(model, i, j, t):
            return model.v[j,t] == model.v[i,t] - 2*(model.R[i,j]*model.p[i,j,t] + model.X[i,j]*model.q[i,j,t])
        model.lpfe_r = py.Constraint(model.L, model.T, rule=LPFE_R)

        def DP_init(model, i, t):
            return model.dp[i,t] == model.pv[i,t] - model.pl[i,t] + model.ds[i,t] - model.ch[i,t]
        model.dp_init = py.Constraint(model.A, model.T, rule=DP_init)

        def DP_init2(model, i, t):
            return model.dp[i,t] == model.dp_pos[i,t] - model.dp_neg[i,t]
        model.dp_init2 = py.Constraint(model.A, model.T, rule=DP_init2)

        model.bus_uno_buy  = Constraint(model.B, model.T, rule=lambda m,i,t: m.kappa_bg[i,t] <= m.P_BG_max[i])
        model.bus_uno_sell = Constraint(model.B, model.T, rule=lambda m,i,t: m.kappa_sg[i,t] <= m.P_SG_max[i])

        model.DP_Pos  = py.Constraint(model.A, model.T,
            rule=lambda m,i,t: m.dp_pos[i,t] <= (m.PV_max[i,t] * m.DG[i] * 200) / m.sb * m.y[i,t])
        model.DP_Neg  = py.Constraint(model.A, model.T,
            rule=lambda m,i,t: m.dp_neg[i,t] <= m.M * (1 - m.y[i,t]))

        def SOCR(model, i, t):
            if t == 1:
                return model.soc[i,t] == 0.5 * (model.BT[i] / model.sb)
            return model.soc[i,t] == model.soc[i,t-1] + model.Fi*model.ch[i,t] - model.ds[i,t]/model.Fi
        model.socr = Constraint(model.A, model.T, rule=SOCR)

        model.chr   = Constraint(model.A, model.T, rule=lambda m,i,t: m.ch[i,t] <= (m.PB/m.sb)*m.w[i,t])
        model.dsr   = Constraint(model.A, model.T, rule=lambda m,i,t: m.ds[i,t] <= (m.PB/m.sb)*(1-m.w[i,t]))
        model.dsr_w = Constraint(model.A, model.T, rule=lambda m,i,t: m.w[i,t] <= m.E_BT[i])

        model.DP_Pos_2 = py.Constraint(model.A, model.T,
            rule=lambda m,i,t: m.dp_pos[i,t] == m.p_sg[i,t] + m.p_sm[i,t])
        model.DP_Neg_2 = py.Constraint(model.A, model.T,
            rule=lambda m,i,t: m.dp_neg[i,t] == m.p_bg[i,t] + m.p_bm[i,t])
        model.market_balance = py.Constraint(model.T,
            rule=lambda m,t: sum(m.p_sm[i,t] for i in m.A) == sum(m.p_bm[i,t] for i in m.A))
        model.kappa_bg_balance = py.Constraint(model.T,
            rule=lambda m,t: sum(m.p_bg[i,t] for i in m.A) == sum(m.kappa_bg[i,t] for i in m.B))
        model.kappa_sg_balance = py.Constraint(model.T,
            rule=lambda m,t: sum(m.p_sg[i,t] for i in m.A) == sum(m.kappa_sg[i,t] for i in m.B))
        model.QG_max = py.Constraint(model.B, model.T, rule=lambda m,i,t: m.qg[i,t] <= m.QGmax[i])
        model.QG_min = py.Constraint(model.B, model.T, rule=lambda m,i,t: m.qg[i,t] >= m.QGmin[i])
        model.V_max  = py.Constraint(model.B, model.T, rule=lambda m,i,t: m.v[i,t] <= m.Vmax[i]**2)
        model.V_min  = py.Constraint(model.B, model.T, rule=lambda m,i,t: m.v[i,t] >= m.Vmin[i]**2)
        model.PV_gen = py.Constraint(model.A, model.T,
            rule=lambda m,i,t: m.pv[i,t] <= m.PV_max[i,t] * m.DG[i] / m.sb)
        model.PL_restriction = py.Constraint(model.A, model.T,
            rule=lambda m,i,t: m.pl[i,t] == m.PL_Mu[i,t] / m.sb)
        model.Smax = py.Constraint(model.L, model.T, model.RR,
            rule=lambda m,i,j,t,r: m.AA[r]*m.p[i,j,t] + m.BB[r]*m.q[i,j,t] + m.CC[r]*(m.s_max[i,j]/m.sb) <= 0)
        model.Limit_SOC    = Constraint(model.A, model.T,
            rule=lambda m,i,t: m.soc[i,t] >= (m.BT[i]/m.sb)*m.SOC_min)
        model.Limit_SOCsup = Constraint(model.A, model.T,
            rule=lambda m,i,t: m.soc[i,t] <= (m.BT[i]/m.sb)*m.SOC_max)

        return model.create_instance()

    def ExportResults(self, model, filename):
        factor_pot = self.MWtoKw * self.sb

        def get_var_df(variable, scale=1.0):
            data = {k: value(v) for k, v in variable.items()}
            if not data:
                return pd.DataFrame()
            df = pd.Series(data).unstack(level=-1) * scale
            df = df.sort_index(axis=1).sort_index()
            df.index.name = 'Bus'
            return df.reset_index()

        dfs = {
            'PV':      get_var_df(model.pv,       factor_pot),
            'pl':      get_var_df(model.pl,       factor_pot),
            'DP':      get_var_df(model.dp,       factor_pot),
            'DP_pos':  get_var_df(model.dp_pos,   factor_pot),
            'DP_neg':  get_var_df(model.dp_neg,   factor_pot),
            'soc':     get_var_df(model.soc,      factor_pot),
            'CH':      get_var_df(model.ch,       factor_pot),
            'DS':      get_var_df(model.ds,       factor_pot),
            'P_SG':    get_var_df(model.p_sg,     factor_pot),
            'P_SM':    get_var_df(model.p_sm,     factor_pot),
            'P_BG':    get_var_df(model.p_bg,     factor_pot),
            'P_BM':    get_var_df(model.p_bm,     factor_pot),
            'Kappa_BG':get_var_df(model.kappa_bg, factor_pot),
            'Kappa_SG':get_var_df(model.kappa_sg, factor_pot),
            'y':       get_var_df(model.y,        1.0),
            'w':       get_var_df(model.w,        1.0),
        }

        summary_keys = ['PV','pl','DP','DP_pos','DP_neg','soc','CH','DS','P_SG','P_SM','P_BG','P_BM']
        frames = []
        for k in summary_keys:
            df = dfs[k]
            if df.empty: continue
            cols = [c for c in df.columns if c != 'Bus']
            frames.append(pd.DataFrame({'Bus': df['Bus'], k: df[cols].sum(axis=1)}))
        summary = frames[0]
        for f in frames[1:]:
            summary = pd.merge(summary, f, on='Bus', how='outer')

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            summary.to_excel(writer, sheet_name='Resumen', index=False)
            for sheet, df in dfs.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet, index=False)


if __name__ == '__main__':
    ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE  = os.path.join(ROOT, 'data', 'Case_Study', 'Case_Study_365.xlsx')
    OUTPUT_DIR = os.path.join(ROOT, 'results', 'deterministic_365')

    running = P2P_Model_Deterministic()
    running.ReadExcelFile(DATA_FILE)
    running.Run365(output_dir=OUTPUT_DIR)
    print(f"\nDone. Results saved to: {OUTPUT_DIR}")
