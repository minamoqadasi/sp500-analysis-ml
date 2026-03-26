import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data
sp500 = yf.download("^GSPC", start="2022-01-01")

sp500.columns = sp500.columns.get_level_values(0)
sp500 = sp500[['Close']].rename(columns={'Close': 'Price'})
sp500['Price'] = sp500['Price'].squeeze()

# Return
sp500['Return'] = sp500['Price'].pct_change()
# Average
sp500['MA_20'] = sp500['Price'].rolling(20).mean() / sp500['Price']
sp500['MA_50'] = sp500['Price'].rolling(50).mean() / sp500['Price']
# Volatility
sp500['Volatility'] = sp500['Return'].rolling(20).std()
# Momentum
# Make sure both are Series
sp500['Momentum'] = sp500['Price'] - sp500['Price'].rolling(20).mean()
# Target
sp500['Target'] = (sp500['Return'].shift(-1) > 0).astype(int)

sp500 = sp500.dropna()

# Test
X = sp500[['Return', 'MA_20', 'MA_50', 'Volatility', 'Momentum']]
y = sp500['Target']

split = int(len(sp500) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))

# Backtest
sp500_test = sp500.iloc[split:].copy()

sp500_test['Prediction'] = preds

# Strategy only return when output is 1
sp500_test['Strategy_Return'] = sp500_test['Return'] * sp500_test['Prediction']

# Cumulative returns
sp500_test['Market_Return'] = (1 + sp500_test['Return']).cumprod()
sp500_test['Strategy_Growth'] = (1 + sp500_test['Strategy_Return']).cumprod()

# Results
plt.figure(figsize=(10, 5))
plt.plot(sp500_test['Market_Return'], label='Market')
plt.plot(sp500_test['Strategy_Growth'], label='Strategy')
plt.legend()
plt.title("Strategy vs Market Performance")
plt.xlabel("Date")
plt.ylabel("Growth")
plt.show()

