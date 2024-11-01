import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
# Memuat dataset
data = pd.read_csv('weather.csv')

# Memeriksa nilai NaN dan kolom yang ada
print("Jumlah nilai NaN sebelum pembersihan:")
print(data.isnull().sum())
print("\nKolom yang tersedia dalam dataset:")
print(data.columns)

# Mengisi nilai NaN pada kolom numerik dengan rata-rata kolom
data_numeric = data.select_dtypes(include=['number'])  # Memilih kolom numerik
data[data_numeric.columns] = data_numeric.fillna(data_numeric.mean())

# Mengisi nilai NaN pada kolom non-numerik, jika ada
data['WindGustDir'] = data['WindGustDir'].fillna(data['WindGustDir'].mode()[0])  # Mengisi dengan modus
data['WindDir9am'] = data['WindDir9am'].fillna(data['WindDir9am'].mode()[0])  # Mengisi dengan modus
data['WindGustSpeed'] = data['WindGustSpeed'].fillna(data['WindGustSpeed'].mean())  # Mengisi dengan rata-rata
data['WindSpeed9am'] = data['WindSpeed9am'].fillna(data['WindSpeed9am'].mean())  # Mengisi dengan rata-rata

# Mengisi nilai NaN pada kolom WindDir3pm
data['WindDir3pm'] = data['WindDir3pm'].fillna(data['WindDir3pm'].mode()[0])  # Mengisi dengan modus

# Periksa kembali jumlah nilai NaN setelah pengisian
print("\nJumlah nilai NaN setelah pembersihan:")
print(data.isnull().sum())

# Menghapus baris yang masih memiliki nilai NaN setelah pengisian
data = data.dropna()

# Memisahkan fitur dan target setelah pembersihan
# Periksa kolom yang tersedia dan sesuaikan dengan yang ada
X_rainfall = data[['Temp9am', 'Temp3pm', 'Pressure9am', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']]
y_rainfall = data['Rainfall']

X_evaporation = data[['Temp9am', 'Temp3pm', 'Rainfall', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']]
y_evaporation = data['Evaporation']

X_sunshine = data[['Temp9am', 'Temp3pm', 'Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']]
y_sunshine = data['Sunshine']

# Memisahkan data menjadi data latih dan data uji
X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(X_rainfall, y_rainfall, test_size=0.2, random_state=42)
X_train_evap, X_test_evap, y_train_evap, y_test_evap = train_test_split(X_evaporation, y_evaporation, test_size=0.2, random_state=42)
X_train_sun, X_test_sun, y_train_sun, y_test_sun = train_test_split(X_sunshine, y_sunshine, test_size=0.2, random_state=42)

# Membuat model regresi linier
model_rain = LinearRegression()
model_evap = LinearRegression()
model_sun = LinearRegression()

# Melatih model
model_rain.fit(X_train_rain, y_train_rain)
model_evap.fit(X_train_evap, y_train_evap)
model_sun.fit(X_train_sun, y_train_sun)

# Menyimpan model ke file
joblib.dump(model_rain, 'rainfall_model.pkl')
joblib.dump(model_evap, 'evaporation_model.pkl')
joblib.dump(model_sun, 'sunshine_model.pkl')

# Menampilkan bentuk data latih
print("Jumlah shape Rainfall:", X_train_rain.shape)  # Harusnya (jumlah_sampel, 7)
print("Jumlah shape Evaporation:", X_train_evap.shape)  # Harusnya (jumlah_sampel, 7)
print("Jumlah shape Sunshine:", X_train_sun.shape)  # Harusnya (jumlah_sampel, 8 jika tidak ada kolom yang dihapus)
