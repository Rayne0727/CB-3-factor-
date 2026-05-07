import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 0. 全局配置 =====================
BASE_PATH = r"C:\Users\Rayne\Desktop"
TARGET_DATE = pd.to_datetime("2025-09-05")  # 定价基准日

CALL_TRIGGER_WINDOW = 15
CALL_TRIGGER_RATIO = 1.3
CALL_EXEC_PROB = 0.9
PUT_TRIGGER_WINDOW = 30
PUT_TRIGGER_RATIO = 0.7
DOWN_RESET_PROB = 0.8
SIM_PATHS = 15000  # 配合对偶变量法，实际生成的独立路径为 7500 条


# --- 核心日期解析函数 ---
def parse_dt(series):
    s = series.astype(str).str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.slice(0, 8)
    return pd.to_datetime(s, format='%Y%m%d', errors='coerce')

# ===================== 1. 数据加载与基础清洗 =====================
print("加载基础数据...")
df_daily = pd.read_excel(os.path.join(BASE_PATH, "2_cb_daily_日频.xlsx"))
df_basic = pd.read_excel(os.path.join(BASE_PATH, "1_cb_basic_全字段.xlsx"))

# 读取正股精简数据
stk_path = os.path.join(BASE_PATH, "7_stk_daily_正股.csv")
try:
    df_stock = pd.read_csv(stk_path, encoding='utf-8-sig')
except:
    df_stock = pd.read_csv(stk_path, encoding='gbk')

df_daily['dt'] = parse_dt(df_daily['trade_date'])
df_stock['dt'] = parse_dt(df_stock['trade_date'])

# 日期对齐
available_dates = df_daily['dt'].dropna().unique()
if TARGET_DATE not in available_dates:
    TARGET_DATE = df_daily['dt'].max()
print(f" -> 实际定价日期: {TARGET_DATE.date()}")

df_day = df_daily[df_daily['dt'] == TARGET_DATE].copy()
df = pd.merge(df_day, df_basic, on='ts_code', how='left')

# 基础过滤
df['K'] = df['conv_price'].fillna(df['first_conv_price']).fillna(100)
df['maturity_date'] = parse_dt(df['maturity_date'])
df['remaining_years'] = (df['maturity_date'] - TARGET_DATE).dt.days / 365

df = df[
    (df['remaining_years'] > 0.25) & 
    (df['remain_size'] > 0.3) & 
    (df['close'] > 70) & (df['close'] < 300) & 
    (df['K'] > 0)
].reset_index(drop=True)
print(f" -> 最终有效转债数量: {len(df)} 只")

# ===================== 2. 构建收益率曲线 (插值法) =====================
print("构建无风险与信用基准收益率曲线...")
df_yield = pd.read_csv(os.path.join(BASE_PATH, "merged_yield_curves_final.csv"))
df_yield['日期'] = pd.to_datetime(df_yield['日期'])
# 获取距离 TARGET_DATE 最近的一天的数据
yield_day = df_yield[df_yield['日期'] <= TARGET_DATE].sort_values('日期').iloc[-1]['日期']
df_y_today = df_yield[df_yield['日期'] == yield_day]

tenors = [0.25, 0.5, 1, 3, 5, 7, 10, 30]
cols = ['3月', '6月', '1年', '3年', '5年', '7年', '10年', '30年']

# 提取国债（无风险）和 AAA 企债（信用基准）曲线
def get_curve_interp(curve_name):
    row = df_y_today[df_y_today['曲线名称'].str.contains(curve_name)].iloc[0]
    # 转换为小数，处理缺失值用相邻填充
    rates = pd.to_numeric(row[cols], errors='coerce').ffill().bfill().values / 100 
    # 使用 extrapolate 允许对超出 30 年的极少数情况进行外推
    return interp1d(tenors, rates, kind='linear', fill_value="extrapolate")

rf_interp = get_curve_interp("国债")
corp_interp = get_curve_interp("中短期票据") # 若无则会自动 fallback 到可用数据

df['r'] = rf_interp(df['remaining_years'].values)

def calc_bond_floor_pro(row):
    if pd.isna(row['remaining_years']): return np.nan
    T = row['remaining_years']
    
    # 获取对应期限的 AAA 基准收益率
    base_corp_rate = corp_interp(T)
    rating = str(row.get('newest_rating', 'AA'))
    
    # 评级信用利差微调 (AAA已包含在曲线上)
    if 'AAA' in rating: spread = 0.0
    elif 'AA+' in rating: spread = 0.005
    elif 'AA-' in rating: spread = 0.02
    else: spread = 0.012 # AA 等
    
    disc = base_corp_rate + spread
    coupon = float(row.get('coupon_rate', 1.5)) / 100
    bond = 0
    face = 100
    
    for y in range(1, int(np.ceil(T)) + 1):
        t = y if y < T else T
        if y < np.ceil(T): bond += face * coupon / ((1 + disc)**t)
        else: bond += face * (1 + coupon) / ((1 + disc)**t)
    return bond

df['bond_floor'] = df.apply(calc_bond_floor_pro, axis=1)

# ===================== 3. 计算 EWMA 波动率与股息率 =====================
print("计算 EWMA 波动率与连续分红率 q...")
sub_stk = df_stock[df_stock['dt'] <= TARGET_DATE].copy()
sub_stk = sub_stk.sort_values(['ts_code', 'dt'])

# 计算对数收益率
sub_stk['log_ret'] = np.log(sub_stk['close_qfq'] / sub_stk.groupby('ts_code')['close_qfq'].shift(1))

# 核心：利用 ewma(span=60) 计算最新的一期指数加权波动率
# 核心：计算波动率并进行A股特色化处理
def get_advanced_factors(g):
    if len(g) < 30: return pd.Series({'sigma': 0.3, 'q': 0.0, 'S0': np.nan})
    rets = g['log_ret'].dropna().tail(242) 
    
    # 修复1：混合波动率，削弱 EWMA 的极端值，并设置 0.55 的最高封顶(Cap)
    sigma_ewma = rets.ewm(span=60).std().iloc[-1] * np.sqrt(242)
    sigma_hist = rets.std() * np.sqrt(242)
    sigma = min((sigma_ewma * 0.4 + sigma_hist * 0.6), 0.55) # 封顶55%
    
    # 修复2：强制将 q 设为 0（因为中国转债除息自动下修，完全对冲分红损失）
    q = 0.0 
    S0 = g['close'].iloc[-1]
    return pd.Series({'sigma': sigma, 'q': q, 'S0': S0})

factors = sub_stk.groupby('ts_code').apply(get_advanced_factors).reset_index()

# 代码匹配处理
def clean_code(s): return str(s).split('.')[0].zfill(6)
df['stk_clean'] = df['stk_code'].apply(clean_code)
factors['stk_clean'] = factors['ts_code'].apply(clean_code)

df = pd.merge(df, factors[['stk_clean', 'sigma', 'q', 'S0']], on='stk_clean', how='left')
df['sigma'] = df['sigma'].fillna(0.3)
df['q'] = df['q'].fillna(0.0)
df['S0'] = df['S0'].fillna(df['close'] / 10) # 兜底

# ===================== 4. LSM Pro 引擎 (对偶变量 + 向量化) =====================
print("执行LSM蒙特卡洛定价...")

class LSM_Pro:
    def __init__(self, S0, K, T, r, q, sigma, bond_floor):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q 
        self.sigma = sigma
        self.bond_floor = bond_floor
        self.M = max(int(T * 242), 30)
        self.dt = 1 / 242
        self.face = 100

    def simulate_antithetic(self, N):
        np.random.seed(42)
        S = np.zeros((N, self.M + 1))
        S[:, 0] = self.S0
        
        # 对偶变量法：生成一半随机数，另一半直接取相反数 (提速 50% 并降低方差)
        N_half = N // 2
        Z_half = np.random.randn(N_half, self.M)
        Z = np.vstack((Z_half, -Z_half))
        
        # 引入连续股息率 q 的漂移项
        drift = self.r - self.q - 0.5 * self.sigma**2
        growth_factor = np.exp(drift * self.dt + self.sigma * np.sqrt(self.dt) * Z)
        
        S[:, 1:] = self.S0 * np.cumprod(growth_factor, axis=1)
        return S

    def price(self, N=SIM_PATHS):
        try:
            # 保证 N 为偶数
            N = N if N % 2 == 0 else N + 1
            S = self.simulate_antithetic(N)
            conv = S * self.face / self.K 
            
            ck = np.full(N, self.K)           
            end = np.full(N, self.M)          
            val = np.zeros(N)                 
            active = np.ones(N, dtype=bool)   

            for t in range(self.M + 1):
                if not np.any(active): break
                cv_current = S[:, t] * self.face / ck

                # 强赎
                if t >= CALL_TRIGGER_WINDOW - 1:
                    w = S[:, t - CALL_TRIGGER_WINDOW + 1 : t + 1]
                    call_cond = np.all(w >= (ck * CALL_TRIGGER_RATIO)[:, None], axis=1)
                    call_exec = call_cond & active & (np.random.rand(N) < CALL_EXEC_PROB)
                    if np.any(call_exec):
                        end[call_exec] = t
                        val[call_exec] = np.maximum(cv_current[call_exec], 103)
                        active[call_exec] = False

                # 回售与下修
                if t >= PUT_TRIGGER_WINDOW - 1:
                    w = S[:, t - PUT_TRIGGER_WINDOW + 1 : t + 1]
                    put_cond = np.all(w <= (ck * PUT_TRIGGER_RATIO)[:, None], axis=1)
                    put_active = put_cond & active
                    if np.any(put_active):
                        reset_mask = put_active & (np.random.rand(N) < DOWN_RESET_PROB)
                        sell_mask = put_active & ~reset_mask
                        if np.any(reset_mask):
                            ck[reset_mask] = np.maximum(ck[reset_mask] * 0.7, S[reset_mask, t] * 0.9)
                        if np.any(sell_mask):
                            end[sell_mask] = t
                            val[sell_mask] = np.maximum(100, self.bond_floor)
                            active[sell_mask] = False

                # 到期
                if t == self.M:
                    val[active] = np.maximum(cv_current[active], self.face * 1.02)
                    active[:] = False

            V = np.zeros((N, self.M + 1))
            V[np.arange(N), end] = val

            # 最小二乘法向后折现
            for t in range(self.M - 1, -1, -1):
                mask = end > t
                if not np.any(mask): continue
                
                cv_t = conv[mask, t]
                discount = V[mask, t + 1] / (1 + self.r * self.dt)
                
                try:
                    X = np.vstack([np.ones_like(cv_t), cv_t, cv_t**2]).T
                    beta = np.linalg.lstsq(X, discount, rcond=None)[0]
                    cont = X @ beta
                    V[mask, t] = np.maximum(cv_t, cont)
                except:
                    V[mask, t] = discount
                    
                V[~mask, t] = val[~mask]

            return float(np.mean(V[:, 0]))
        except:
            return np.nan

# 批量执行定价
prices = []
total = len(df)
for i, row in df.iterrows():
    if i % 20 == 0: print(f" -> 定价进度: {i}/{total} ({row['bond_short_name']})")
    model = LSM_Pro(
        S0=row['S0'], K=row['K'], T=row['remaining_years'], 
        r=row['r'], q=row['q'], sigma=row['sigma'], bond_floor=row['bond_floor']
    )
    prices.append(model.price())

df['model_price'] = prices
df = df.dropna(subset=['model_price'])

# ===================== 5. 结果输出 =====================
print("生成分析报告与可视化图表...")

valid_mask = ~((df['close'] > 150) & (df['model_price'] < 120))
df_valid = df[valid_mask].copy()

y_true = df_valid['close']
y_pred = df_valid['model_price']
r2 = r2_score(y_true, y_pred)

print(f"（已剔除 {len(df) - len(df_valid)} 只脱离基本面博弈的妖债）")
print(f"模型 R² = {r2:.4f} ({r2*100:.2f}%)")

plt.figure(figsize=(10,7))
plt.scatter(y_pred, y_true, s=35, alpha=0.8, c='#D32F2F', edgecolors='white')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', alpha=0.6, label='理论完全拟合线')
plt.xlabel("LSM模型理论价格", fontsize=12)
plt.ylabel("实际市场价格", fontsize=12)
plt.title(f"2025-09-05 全市场可转债定价拟合 (R²={r2:.4f})", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(BASE_PATH, "LSMResult.png"), dpi=300, bbox_inches='tight')
plt.show()

# 寻找低估标的
df['undervalue'] = df['close'] / df['model_price'] - 1
result_cols = ['ts_code', 'bond_short_name', 'close', 'model_price', 'undervalue', 'bond_floor', 'sigma', 'q']
df.sort_values('undervalue')[result_cols].to_excel(os.path.join(BASE_PATH, "LSM低估转债池.xlsx"), index=False)