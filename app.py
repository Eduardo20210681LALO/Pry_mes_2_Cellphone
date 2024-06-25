from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('model1.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        resoloution = float(request.form['resoloution'])
        ppi = float(request.form['ppi'])
        ram = float(request.form['ram'])

        Front_Cam = float(request.form['Front_Cam'])
        battery = float(request.form['battery'])
        thickness = float(request.form['thickness'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[resoloution, ppi, ram, Front_Cam, battery, thickness]], columns=['resoloution', 'ppi', 'ram', 'Front_Cam', 'battery', 'thickness'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Realizar predicciones
        # Realizar predicciones
        prediction = model.predict(data_df)
        predicted_class = int(prediction[0])  # Convertir la predicción a entero

        
        app.logger.debug(f'Predicción: {predicted_class}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': predicted_class})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': 'Error en la solicitud. Detalles en el registro del servidor.'}), 400

if __name__ == '__main__':
    app.run(debug=True)

