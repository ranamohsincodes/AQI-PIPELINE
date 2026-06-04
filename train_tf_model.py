import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import joblib

# Load your existing clean dataset
df = pd.read_csv('karachi_clean_dataset.csv')
df = df.dropna()

features = ['pm25','pm10','no2','co','o3','so2','nh3',
            'hour','day_of_week','month','aqi_lag_1h','aqi_lag_3h','aqi_change']
target = 'aqi'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (required for neural nets, unlike RF)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simple model - don't overcomplicate it
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(13,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate
y_pred = model.predict(X_test_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"TF Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Save
model.save('aqi_tf_model.keras')
joblib.dump(scaler, 'scaler.pkl')
