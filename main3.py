import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
data = pd.read_csv('../data3/energy.csv', parse_dates=['Datetime'], index_col='Datetime')

# Preprocess
data = data.resample('H').mean()
data = data.fillna(method='ffill')

# Features
data['hour'] = data.index.hour
data['day'] = data.index.dayofweek

# Split
X = data[['hour', 'day']]
y = data['Energy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=500)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)

# Save model
joblib.dump(model, '../model3/energy_model.pkl')

# Graph
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Energy")

plt.savefig('../output3/actual_vs_pred.png')
plt.show()