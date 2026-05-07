import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_PATH = r"C:\Users\Rayne\Desktop"
TARGET_DATE = pd.to_datetime("2025-09-05")

# ===================== 1. 基础数据加载 =====================
print("加载基础数据...")
def parse_dt(series):
    s = series.astype(str).str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.slice(0, 8)
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

df_basic = pd.read_excel(os.path.join(BASE_PATH, "1_cb_basic_全字段.xlsx"))
df_daily = pd.read_excel(os.path.join(BASE_PATH, "2_cb_daily_日频.xlsx"))
df_stk = pd.read_csv(os.path.join(BASE_PATH, "7_stk_daily_正股.csv"), encoding='utf-8-sig')

df_daily['dt'] = parse_dt(df_daily['trade_date'])
df_stk['dt'] = parse_dt(df_stk['trade_date'])

df_day = df_daily[df_daily['dt'] == TARGET_DATE].copy()
df = pd.merge(df_day, df_basic, on='ts_code', how='left')

df['K'] = df['conv_price'].fillna(df['first_conv_price']).fillna(100)
df['maturity_date'] = parse_dt(df['maturity_date'])
df['remaining_years'] = (df['maturity_date'] - TARGET_DATE).dt.days / 365

# 基础过滤：剩余期限>0.25年，余额>0.3亿，
df = df[(df['remaining_years'] > 0.25) & (df['remain_size'] >30000000 ) & (df['K'] > 0)].reset_index(drop=True)
df = df[~((df['close'] > 150) & (df['cb_value'] < 100))].reset_index(drop=True)

# ===================== 2. 构建真实收益率曲线 =====================
df_yield = pd.read_csv(os.path.join(BASE_PATH, "merged_yield_curves_final.csv"))
df_yield['日期'] = pd.to_datetime(df_yield['日期'])
yield_day = df_yield[df_yield['日期'] <= TARGET_DATE].sort_values('日期').iloc[-1]['日期']
df_y_today = df_yield[df_yield['日期'] == yield_day]

tenors = [0.25, 0.5, 1, 3, 5, 7, 10, 30]
cols = ['3月', '6月', '1年', '3年', '5年', '7年', '10年', '30年']

def get_curve_interp(curve_name):
    row = df_y_today[df_y_today['曲线名称'].str.contains(curve_name)].iloc[0]
    rates = pd.to_numeric(row[cols], errors='coerce').ffill().bfill().values / 100 
    return interp1d(tenors, rates, kind='linear', fill_value="extrapolate")

# 国债曲线 (计算 r) 和 中短期票据曲线 (计算债底贴现率)
rf_interp = get_curve_interp("国债")
corp_interp = get_curve_interp("中短期票据")

df['r'] = rf_interp(df['remaining_years'].values)

def calc_bond_floor_precise(row):
    if pd.isna(row['remaining_years']): return np.nan
    T = row['remaining_years']
    base_corp_rate = corp_interp(T)
    
    # 评级利差微调
    rating = str(row.get('newest_rating', 'AA'))
    if 'AAA' in rating: spread = 0.0
    elif 'AA+' in rating: spread = 0.005
    elif 'AA-' in rating: spread = 0.02
    else: spread = 0.012 
    
    disc = base_corp_rate + spread
    coupon = float(row.get('coupon_rate', 1.5)) / 100
    bond = 0
    face = 100
    
    # 逐年现金流贴现
    for y in range(1, int(np.ceil(T)) + 1):
        t = y if y < T else T
        if y < np.ceil(T): bond += face * coupon / ((1 + disc)**t)
        else: bond += face * (1 + coupon) / ((1 + disc)**t)
    return bond

df['bond_floor'] = df.apply(calc_bond_floor_precise, axis=1)

# ===================== 3. XGBoost  =====================
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(BASE_PATH, "xgb_call_model.json"))

df_stk_sub = df_stk[df_stk['dt'] <= TARGET_DATE].sort_values(['ts_code', 'dt'])
df_stk_sub['log_ret'] = np.log(df_stk_sub['close_qfq'] / df_stk_sub.groupby('ts_code')['close_qfq'].shift(1))

def get_snapshot_factors(g):
    if len(g) < 60: return pd.Series({'sigma': 0.3, 'q': 0.0, 'pe_ttm': 20, 'pb': 2, 'S0': np.nan})
    rets = g['log_ret'].tail(242)
    sigma = min((rets.ewm(span=60).std().iloc[-1] * 0.4 + rets.std() * 0.6) * np.sqrt(242), 0.55)
    return pd.Series({
        'sigma': sigma, 'q': 0.0,
        'pe_ttm': g['pe_ttm'].iloc[-1], 'pb': g['pb'].iloc[-1],
        'stk_vol_60d': sigma, 'S0': g['close'].iloc[-1]
    })

df['stk_clean'] = df['stk_code'].astype(str).str.split('.').str[0].str.zfill(6)

factors = df_stk_sub.groupby('ts_code').apply(get_snapshot_factors).reset_index()
factors['stk_clean'] = factors['ts_code'].astype(str).str.split('.').str[0].str.zfill(6)

factors = factors.drop(columns=['ts_code']) 

df = pd.merge(df, factors, on='stk_clean', how='left').fillna({'sigma':0.3, 'q':0.0})
df['S0'] = df['S0'].fillna(df['close']/10)

# ===================== 4.  LSM 蒙特卡洛 =====================

class Ultimate_LSM:
    def __init__(self, row, M_days, N_paths):
        self.S0, self.K, self.T = row['S0'], row['K'], row['remaining_years']
        self.r, self.q, self.sigma = row['r'], row['q'], row['sigma']
        self.bond_floor_T0 = row['bond_floor'] 
        self.M = max(int(self.T * 242), 30)
        self.dt = 1 / 242
        self.face = 100
        self.N = N_paths if N_paths % 2 == 0 else N_paths + 1
        
        self.static_feats = {
            'cb_vol_60d': row.get('cb_vol_60d', self.sigma),
            'cb_amt_20d': row.get('amount', 50000),
            'pe_ttm': row.get('pe_ttm', 20),
            'pb': row.get('pb', 2),
            'stk_vol_60d': self.sigma,
            'remain_size': row.get('remain_size', 5)
        }

    def simulate_antithetic(self):
        np.random.seed(42)
        S = np.zeros((self.N, self.M + 1))
        S[:, 0] = self.S0
        Z = np.random.randn(self.N // 2, self.M)
        Z = np.vstack((Z, -Z))
        drift = self.r - self.q - 0.5 * self.sigma**2
        growth = np.exp(drift * self.dt + self.sigma * np.sqrt(self.dt) * Z)
        S[:, 1:] = self.S0 * np.cumprod(growth, axis=1)
        return S

    def price(self):
        try:
            S = self.simulate_antithetic()
            conv = S * self.face / self.K 
            ck = np.full(self.N, self.K)           
            end = np.full(self.N, self.M)          
            val = np.zeros(self.N)                 
            active = np.ones(self.N, dtype=bool)   

            # --- 第一遍：从前向后寻找触发点 (强赎 & 回售)---
            for t in range(self.M + 1):
                if not np.any(active): break
                cv_current = S[:, t] * self.face / ck

                # 1. 强赎触发 
                if t >= 14: 
                    w = S[:, t - 14 : t + 1]
                    trigger_mask = np.all(w >= (ck * 1.3)[:, None], axis=1) & active
                    if np.any(trigger_mask):
                        n_trig = np.sum(trigger_mask)
                        X_pred = np.zeros((n_trig, 10))
                        X_pred[:, 0] = np.maximum(cv_current[trigger_mask], 103)
                        X_pred[:, 1] = 0.01 
                        X_pred[:, 7] = S[trigger_mask, t] / S[trigger_mask, max(0, t-20)] - 1 
                        X_pred[:, 8] = max(0.01, self.T - t * self.dt) 
                        X_pred[:, 2] = self.static_feats['cb_vol_60d']
                        X_pred[:, 3] = self.static_feats['cb_amt_20d']
                        X_pred[:, 4] = self.static_feats['pe_ttm']
                        X_pred[:, 5] = self.static_feats['pb']
                        X_pred[:, 6] = self.static_feats['stk_vol_60d']
                        X_pred[:, 9] = self.static_feats['remain_size']
                        
                        probs = xgb_model.predict_proba(X_pred)[:, 1]
                        call_exec_mask = np.random.rand(n_trig) < probs
                        
                        if np.any(call_exec_mask):
                            global_exec_idx = np.where(trigger_mask)[0][call_exec_mask]
                            end[global_exec_idx] = t
                            val[global_exec_idx] = np.maximum(cv_current[global_exec_idx], 103)
                            active[global_exec_idx] = False

                # 2. 回售触发 
                if t >= 29:
                    w = S[:, t - 29 : t + 1]
                    put_active = np.all(w <= (ck * 0.7)[:, None], axis=1) & active
                    if np.any(put_active):
                        reset_mask = put_active & (np.random.rand(self.N) < 0.8)
                        sell_mask = put_active & ~reset_mask
                        if np.any(reset_mask): 
                            ck[reset_mask] = np.maximum(ck[reset_mask] * 0.7, S[reset_mask, t] * 0.9)
                        if np.any(sell_mask):
                            # 动态债底：估算 t 时刻的债底
                            floor_t = self.bond_floor_T0 * np.exp(self.r * t * self.dt)
                            end[sell_mask] = t
                            val[sell_mask] = np.maximum(100, floor_t)
                            active[sell_mask] = False

                # 3. 到期终值
                if t == self.M:
                    val[active] = np.maximum(cv_current[active], self.face * 1.02)
                    active[:] = False

            # --- 第二遍：LSM 条件折现倒推---
            V = np.zeros((self.N, self.M + 1))
            V[np.arange(self.N), end] = val

            for t in range(self.M - 1, -1, -1):
                mask_alive = end > t # t 时刻仍在存续的路径
                if not np.any(mask_alive): continue
                
                cv_t = conv[mask_alive, t]
                discount = V[mask_alive, t + 1] / (1 + self.r * self.dt)
                
                # 估算 t 时刻债底 (Pull-to-par approximation)
                floor_t = self.bond_floor_T0 * np.exp(self.r * t * self.dt)
                
                # 划分 B 类（无转股潜能）和 C 类（有转股潜能）
                mask_itm = cv_t > floor_t # In-the-money (C类)
                mask_otm = ~mask_itm      # Out-of-the-money (B类)
                
                # B 类直接折现
                if np.any(mask_otm):
                    V[np.where(mask_alive)[0][mask_otm], t] = discount[mask_otm]
                
                # C 类进行最小二乘法回归 
                if np.any(mask_itm):
                    X_itm = cv_t[mask_itm]
                    Y_itm = discount[mask_itm]
                    try:
                        X_reg = np.vstack([np.ones_like(X_itm), X_itm, X_itm**2]).T
                        beta = np.linalg.lstsq(X_reg, Y_itm, rcond=None)[0]
                        cont_value = X_reg @ beta
                        # 期望存续价值与立即转股价值 PK
                        V[np.where(mask_alive)[0][mask_itm], t] = np.maximum(X_itm, cont_value)
                    except:
                        V[np.where(mask_alive)[0][mask_itm], t] = Y_itm

                # 覆盖已终止路径的终值
                V[~mask_alive, t] = val[~mask_alive]

            return float(np.mean(V[:, 0]))
        except Exception as e:
            return np.nan

# ===================== 5. 执行定价与绘图 =====================
prices = []
total = len(df)
for i, row in df.iterrows():
    if i % 20 == 0: print(f" -> 进度: {i}/{total} ({row['bond_short_name']})")
    # 7500 对偶路径 = 15000 次模拟，完全满足高精度要求
    model = Ultimate_LSM(row, M_days=242, N_paths=7500)
    prices.append(model.price())

df['model_price'] = prices
df = df.dropna(subset=['model_price'])

r2 = r2_score(df['close'], df['model_price'])

print(f"模型 R² = {r2:.4f} ({r2*100:.2f}%)")

plt.figure(figsize=(11,8))
plt.scatter(df['model_price'], df['close'], s=40, alpha=0.85, c='#1E88E5', edgecolors='white', linewidths=0.5)
plt.plot([df['close'].min(), df['close'].max()], [df['close'].min(), df['close'].max()], color='#D32F2F', linestyle='--', lw=2, label='完美拟合线 (y=x)')

# 标注偏离度最大的几只债券
df['abs_diff'] = np.abs(df['model_price'] - df['close'])
top_outliers = df.sort_values('abs_diff', ascending=False).head(5)
for _, row in top_outliers.iterrows():
    plt.annotate(row['bond_short_name'], (row['model_price'], row['close']), 
                 xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)

plt.xlabel("LSM 理论价格", fontsize=12)
plt.ylabel("实际市场价格", fontsize=12)
plt.title(f" 2025-09-05 可转债定价 (R²={r2:.4f})", fontsize=15, fontweight='bold')
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')

save_path_img = os.path.join(BASE_PATH, "LSMReplica_.png")
save_path_csv = os.path.join(BASE_PATH, "LSM定价结果_2.xlsx")
plt.savefig(save_path_img, dpi=300, bbox_inches='tight')
plt.show()

# 导出低估组合 
df['低估程度'] = df['close'] / df['model_price'] - 1
export_cols = ['ts_code', 'bond_short_name', 'close', 'model_price', '低估程度', 'bond_floor', 'sigma']
df.sort_values('低估程度')[export_cols].to_excel(save_path_csv, index=False)
print(f"-> 详细定价数据与低估组合已导出至: {save_path_csv}")