# AI-Economic
通膨、房價與青年就業
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. 資料清洗與預處理 (Data Cleaning)
# ==========================================

# 假設你從 AREMOS 匯出的 CSV 檔名為 'aremos_data.csv'
# 欄位包含：date (時間), cpi_food (食物類CPI), house_price (房價指標), wage (青年名目薪資)
try:
    # 台灣資料庫常使用 cp950 或 utf-8 
    df = pd.read_csv('aremos_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('aremos_data.csv', encoding='cp950')

# 清洗步驟：將時間欄位轉換為 pandas datetime 格式，並設為索引
# 備註：若 AREMOS 時間格式為 '2023M01'，需先轉換為 '2023-01-01'
df['date'] = df['date'].astype(str).str.replace('M', '-')
df['date'] = pd.to_datetime(df['date'] + '-01' if 'M' in str(df['date'].iloc[0]) else df['date'])
df.set_index('date', inplace=True)

# 處理缺失值 (時間序列常用前後向填補)
df = df.ffill().bfill()

# 計算衍生指標：實質薪資 (名目薪資 / 總CPI * 100)
# 假設你也有調取總體 CPI 指標 'cpi_total'
df['real_wage'] = (df['wage'] / df['cpi_total']) * 100

# 時間序列平穩化：計算年增率 (YoY) 或一階差分 (取對數後差分)
df_diff = np.log(df[['cpi_food', 'house_price', 'real_wage']]).diff().dropna()


# ==========================================
# 2. 傳統總體經濟分析：向量自我迴歸 (VAR 模型)
# ==========================================

print("--- 正在執行 VAR 模型估計 ---")
# 建立 VAR 模型並依據 AIC 自動選擇最佳落後期數 (Lag)
model = VAR(df_diff)
results = model.fit(maxlags=12, ic='aic')
print(results.summary())

# 繪製脈衝響應圖 (Impulse Response Function, IRF)
# 觀察房價 (house_price) 衝擊對實質薪資 (real_wage) 的跨期擠壓效應
irf = results.irf(12)
irf.plot(impulse='house_price', response='real_wage')
plt.title("Impulse Response: House Price Shock on Real Wage")
plt.show()


# ==========================================
# 3. AI/機器學習分析：隨機森林特徵重要性 (Feature Importance)
# ==========================================

print("--- 正在執行 AI 機器學習特徵權重分析 ---")
# 建立滯後特徵 (Lagged Features) 作為模型輸入，模擬時間傳導
for lag in range(1, 4):
    df_diff[f'cpi_food_lag{lag}'] = df_diff['cpi_food'].shift(lag)
    df_diff[f'house_price_lag{lag}'] = df_diff['house_price'].shift(lag)

df_ml = df_diff.dropna()

# 定義特徵 (X) 與目標變數 (y: 實質薪資的變動)
X = df_ml.filter(regex='lag')
y = df_ml['real_wage']

# 訓練隨機森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 視覺化特徵重要性 (哪些指標、落後幾期對實質薪資影響最大)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
plt.title("AI Feature Importance: What Squeezes Youth Real Wage Most?")
plt.xlabel("Importance Score")
plt.ylabel("Macroeconomic Indicators (Lagged)")
plt.tight_layout()
plt.show()
