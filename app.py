from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Memuat model dan scaler
model = joblib.load('model/rul_predictor_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Membaca data yang telah dinormalisasi untuk mendapatkan range normal
df_normalized = pd.read_csv('data_normalized.csv')
columns = df_normalized.columns.tolist()

# Menghitung nilai minimum dan maksimum untuk setiap kolom
range_normal = {col: (df_normalized[col].min(), df_normalized[col].max()) for col in columns}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Mengambil data dari form
        input_data = {col: [float(request.form[col])] for col in columns}
        input_df = pd.DataFrame(input_data)
        
        # Melakukan normalisasi
        input_normalized = pd.DataFrame(scaler.transform(input_df), columns=columns)
        
        # Melakukan prediksi
        pred = model.predict(input_normalized)
        
        # Mengubah prediksi menjadi bilangan bulat dan menambahkan "hari"
        pred_int = int(pred[0])
        prediction = f"{pred_int} hari"
        
    return render_template('index.html', columns=columns, range_normal=range_normal, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
