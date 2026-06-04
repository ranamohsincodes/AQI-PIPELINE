import pandas as pd
import requests
import os
from datetime import datetime

# =========================
# CONFIG
# =========================
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY")
LAT = 24.8607
LON = 67.0011
CITY = "Karachi"
RAW_FILE = "karachi_raw_data.csv"
FINAL_FILE = "karachi_clean_dataset.csv"

# =========================
# FETCH WEATHER (temp, humidity, pressure, wind)
# =========================
def fetch_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()
    if "main" not in data:
        print("Weather Error:", data)
        return None
    return {
        "temperature": data["main"]["temp"],
        "humidity":    data["main"]["humidity"],
        "pressure":    data["main"]["pressure"],
        "wind_speed":  data["wind"]["speed"],
    }

# =========================
# FETCH AQI + POLLUTANTS (OpenWeather Air Pollution API)
# =========================
def fetch_aqi():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
    res = requests.get(url)
    data = res.json()
    if "list" not in data:
        print("AQI Error:", data)
        return None
    d = data["list"][0]
    return {
        "aqi":  d["main"]["aqi"],
        "pm25": d["components"]["pm2_5"],
        "pm10": d["components"]["pm10"],
        "no2":  d["components"]["no2"],
        "co":   d["components"]["co"],
        "o3":   d["components"]["o3"],
        "so2":  d["components"]["so2"],
        "nh3":  d["components"]["nh3"],
    }

# =========================
# COLLECT ONE SAMPLE
# =========================
def collect_sample():
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    weather = fetch_weather()
    aqi     = fetch_aqi()

    if not weather or not aqi:
        print(f"[{timestamp}] Skipped — API error")
        return

    row = {
        "timestamp":   timestamp,
        "city":        CITY,
        "aqi":         aqi["aqi"],
        "pm25":        aqi["pm25"],
        "pm10":        aqi["pm10"],
        "no2":         aqi["no2"],
        "co":          aqi["co"],
        "o3":          aqi["o3"],
        "so2":         aqi["so2"],
        "nh3":         aqi["nh3"],
        "temperature": weather["temperature"],
        "humidity":    weather["humidity"],
        "pressure":    weather["pressure"],
        "wind_speed":  weather["wind_speed"],
    }

    df = pd.DataFrame([row])
    if os.path.exists(RAW_FILE):
        df.to_csv(RAW_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RAW_FILE, index=False)

    print(f"[{timestamp}] Saved — AQI: {aqi['aqi']}, PM2.5: {aqi['pm25']}, Temp: {weather['temperature']}°C")

# =========================
# FEATURE ENGINEERING
# =========================
def build_clean_dataset():
    if not os.path.exists(RAW_FILE):
        print("No raw data found.")
        return

    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.strip().str.lower()
    df.columns = df.columns.str.strip().str.lower()
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.dropna(subset=["aqi"])

    # Time features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month

    # Lag features
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)

    # AQI change rate
    df["aqi_change"] = df["aqi"].diff()

    df = df.dropna(subset=["aqi", "aqi_lag_1h", "aqi_lag_3h", "aqi_change"])
    df.to_csv(FINAL_FILE, index=False)
    print(f"\nClean dataset saved: {len(df)} rows")
    print(df.tail(3))

# =========================
# RUN
# =========================
if __name__ == "__main__":
    collect_sample()
    build_clean_dataset()