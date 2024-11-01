from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib  # Untuk memuat model ML yang sudah dilatih


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Fungsi untuk membuat database dan tabel
def create_database():
    db_file = 'weather_predictions.db'
    
    if not os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT,
                temperature REAL,
                pressure REAL,
                humidity REAL,
                wind_speed REAL,
                wind_direction TEXT,
                weather_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                rainfall REAL, 
                evaporation REAL, 
                sunshine REAL
            )
        ''')
        
        conn.commit()
        conn.close()

# Load datasets
weather_data = pd.read_csv('weather.csv')
weather_history = pd.read_csv('weatherHistory.csv')
weather_custom = pd.read_csv('data_cuaca.csv')

# Mengubah kolom timestamp menjadi format datetime
weather_history['Formatted Date'] = pd.to_datetime(weather_history['Formatted Date'], utc=True)
weather_data['Date'] = pd.date_range(start='1/1/2011', periods=len(weather_data), freq='D')
weather_custom['hpwren_timestamp'] = pd.to_datetime(weather_custom['hpwren_timestamp'], format='ISO8601')

# Mengubah timezone 'UTC' menjadi 'local timezone' jika perlu
weather_history['Formatted Date'] = weather_history['Formatted Date'].dt.tz_convert(None)
weather_data['Date'] = weather_data['Date'].dt.tz_localize(None)
weather_custom['hpwren_timestamp'] = weather_custom['hpwren_timestamp'].dt.tz_localize(None)

# Gabungkan dataset
merged_data = pd.merge(weather_data, weather_history, left_on='Date', right_on='Formatted Date', how='inner')
merged_data = pd.merge(merged_data, weather_custom, left_on='Date', right_on='hpwren_timestamp', how='inner')

# Preprocessing
merged_data.fillna(method='ffill', inplace=True)
merged_data = pd.get_dummies(merged_data, columns=['WindDir9am', 'WindDir3pm'], drop_first=True)

# Normalisasi
features = ['WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
            'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
            'Humidity', 'Wind Speed (km/h)', 'Temperature (C)', 'Apparent Temperature (C)', 'Pressure (millibars)',
            'air_pressure', 'air_temp', 'avg_wind_speed', 'max_wind_speed', 'min_wind_speed', 'relative_humidity']

scaler = StandardScaler()
merged_data[features] = scaler.fit_transform(merged_data[features])

# Model
X = merged_data[features]
y = merged_data['RainToday'].map({'Yes': 1, 'No': 0})
model = RandomForestClassifier()
model.fit(X, y)

# Inisialisasi list untuk menyimpan prediksi
predictions = []

@app.route('/')
def index():
    return render_template('index.html', predictions=predictions)

@app.route('/input')
def input_data():
    return render_template('input.html')

def predict_weather_features_rainfall(features):
    # Memuat model yang telah dilatih
    rainfall_model = joblib.load('rainfall_model.pkl')
    evaporation_model = joblib.load('evaporation_model.pkl')
    sunshine_model = joblib.load('sunshine_model.pkl')

    # Memastikan input adalah DataFrame
    if not isinstance(features, pd.DataFrame):
        raise ValueError("Input features should be a pandas DataFrame")

    # Pastikan input memiliki kolom yang tepat
    expected_columns_rainfall = ['Temp9am', 'Temp3pm', 'Pressure9am', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']
    expected_columns_evaporation = ['Temp9am', 'Temp3pm', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']
    expected_columns_sunshine = ['Temp9am', 'Temp3pm', 'Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']

    # Memisahkan fitur untuk masing-masing model
    X_rainfall = features[expected_columns_rainfall]
    rainfall_prediction = rainfall_model.predict(X_rainfall)

    # Tambahkan hasil prediksi rainfall ke DataFrame untuk prediksi evaporation dan sunshine
    features['Rainfall'] = rainfall_prediction

    X_evaporation = features[expected_columns_evaporation]
    evaporation_prediction = evaporation_model.predict(X_evaporation)

    # Tambahkan hasil prediksi evaporation ke DataFrame untuk prediksi sunshine
    features['Evaporation'] = evaporation_prediction

    X_sunshine = features[expected_columns_sunshine]
    sunshine_prediction = sunshine_model.predict(X_sunshine)

    return rainfall_prediction[0], evaporation_prediction[0], sunshine_prediction[0]


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    date = request.form['date']
    time = request.form['time']
    temperature = float(request.form['temperature'])
    pressure = float(request.form['pressure'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])
    wind_direction = request.form['wind_direction']

    # Dummy input untuk prediksi
    user_input = {
        'WindGustSpeed': wind_speed,
        'WindSpeed9am': wind_speed,
        'WindSpeed3pm': wind_speed,
        'Humidity9am': humidity,
        'Humidity3pm': humidity,
        'Pressure9am': pressure,
        'Pressure3pm': pressure,
        'Cloud9am': 0,
        'Cloud3pm': 0,
        'Temp9am': temperature,
        'Temp3pm': temperature,
        'Humidity': humidity,
        'Wind Speed (km/h)': wind_speed,
        'Temperature (C)': temperature,
        'Apparent Temperature (C)': temperature,
        'Pressure (millibars)': pressure,
        'air_pressure': pressure,
        'air_temp': temperature,
        'avg_wind_speed': wind_speed,
        'max_wind_speed': wind_speed + 0.5,
        'min_wind_speed': wind_speed - 0.5,
        'relative_humidity': humidity
    }

    # Normalisasi input
    user_input_df = pd.DataFrame([user_input])  # Convert to DataFrame
    
    user_input_scaled = scaler.transform(user_input_df)

    # Melakukan prediksi
    prediction = model.predict(user_input_scaled)

    rainfall, evaporation, sunshine = predict_weather_features_rainfall(user_input_df)
    
    # Logika untuk menentukan jenis cuaca
    if prediction[0] == 1:  # Jika ada hujan
        weather_type = 'Hujan' if wind_speed <= 10 else 'Hujan Petir'
    else:
        weather_type = 'Cerah' if humidity <= 80 else 'Mendung'

    # Simpan hasil prediksi ke database
    conn = sqlite3.connect('weather_predictions.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO predictions (location, temperature, pressure, humidity, wind_speed, wind_direction, weather_type, rainfall, evaporation, sunshine)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (location, temperature, pressure, humidity, wind_speed, wind_direction, weather_type, rainfall, evaporation, sunshine))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error while inserting data: {e}")
        conn.rollback()

    # Simpan hasil prediksi ke list
    predictions.append({
        'date': date,
        'location': location,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'weather_type': weather_type,
        'rainfall': rainfall,
        'evaporation': evaporation,
        'sunshine': sunshine,
        'time': time,
    })

    flash('Data berhasil dikirim!')
    return redirect(url_for('result', location=location, date=date, time=time, temperature=temperature, pressure=pressure, humidity=humidity, wind_speed=wind_speed, wind_direction=wind_direction, rainfall=rainfall, evaporation=evaporation, sunshine=sunshine))

@app.route('/result')
def result():
    location = request.args.get('location')
    date = request.args.get('date')
    time = request.args.get('time')

    try:
        temperature = float(request.args.get('temperature'))
        pressure = float(request.args.get('pressure'))
        humidity = float(request.args.get('humidity'))
        wind_speed = float(request.args.get('wind_speed'))
        wind_direction = request.args.get('wind_direction')

        start_time = datetime.strptime(f"{date} {time}", '%Y-%m-%d %H:%M')
        hourly_predictions = []

        for hour_offset in range(24):  # Prediksi setiap jam selama 24 jam
            pred_time = start_time + timedelta(hours=hour_offset)

            # Dummy input untuk prediksi per jam
            hour_input = {
                'WindGustSpeed': wind_speed,
                'WindSpeed9am': wind_speed,
                'WindSpeed3pm': wind_speed,
                'Humidity9am': humidity,
                'Humidity3pm': humidity,
                'Pressure9am': pressure,
                'Pressure3pm': pressure,
                'Cloud9am': 0,
                'Cloud3pm': 0,
                'Temp9am': temperature + hour_offset * 1,
                'Temp3pm': temperature + hour_offset * 1,
                'Humidity': humidity,
                'Wind Speed (km/h)': wind_speed,
                'Temperature (C)': temperature + hour_offset * 1,
                'Apparent Temperature (C)': temperature + hour_offset * 0.3,
                'Pressure (millibars)': pressure,
                'air_pressure': pressure,
                'air_temp': temperature + hour_offset * 0.7,
                'avg_wind_speed': wind_speed,
                'max_wind_speed': wind_speed + 1,
                'min_wind_speed': wind_speed - 1,
                'relative_humidity': humidity
            }

            # Normalisasi input
            hour_input_scaled = scaler.transform(pd.DataFrame([hour_input]))

            # Melakukan prediksi
            prediction = model.predict(hour_input_scaled)

            # Tentukan jenis cuaca
            weather_type = 'Hujan' if prediction[0] == 1 else 'Cerah'

            hourly_predictions.append({'time': pred_time.strftime('%H:%M'), 'weather': weather_type})

        return render_template('result.html', hourly_predictions=hourly_predictions, predictions=predictions, current_date=date, current_time=time)

    except TypeError:
        flash('Data tidak lengkap. Silakan coba lagi.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    create_database()
    app.run(debug=True)
