import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

st.title('Predicción de Mediciones Futuras')
st.subheader('Este proyecto utiliza modelos de aprendizaje automático para predecir la concentración de material particulado PM10 a partir de variables meteorológicas como temperatura, presión atmosférica, radiación solar, velocidad y dirección del viento. Incluye el entrenamiento de modelos, ajuste de hiperparámetros y despliegue de una aplicación interactiva en Streamlit para cargar nuevos datos y obtener predicciones.')

# File upload for the data
uploaded_file = st.file_uploader("Sube tu archivo Excel de mediciones futuras", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        st.subheader('Datos Originales')
        st.dataframe(df.head())
        st.write(df.info())

        # Data Preprocessing
        columns_to_drop = ['Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)', 'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)', 'Precipitación (mm)', 'Humedad Relativa 10m (%)']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        st.subheader('Datos después de eliminar columnas')
        st.dataframe(df.head())
        st.write(df.info())

        # Load the scaler
        try:
            scaler = joblib.load('standard_scaler.pkl')
        except FileNotFoundError:
            st.error("Error: Archivo 'standard_scaler.pkl' no encontrado. Asegúrate de que el archivo esté en '/content/'.")
            st.stop()


        # List of columns to scale
        columns_to_scale = ['Dirección del viento (Grados)', 'Presión atmosférica (mm Hg)', 'Radiación Solar Global (W/m2)', 'Temperatura 10cm (°C)']

        # Apply the loaded scaler to the selected columns
        try:
            df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        except ValueError as e:
             st.error(f"Error al escalar los datos. Asegúrate de que las columnas a escalar existen y tienen los tipos de datos correctos. Detalle: {e}")
             st.stop()

        st.subheader('Datos después de escalar')
        st.dataframe(df.head())


        # Load the model
        try:
            model = joblib.load('best_random_forest_regressor_model_gridsearch.pkl')
        except FileNotFoundError:
            st.error("Error: Archivo 'best_random_forest_regressor_model_gridsearch.pkl' no encontrado. Asegúrate de que el archivo esté en '/content/'.")
            st.stop()


        # Perform predictions
        predictions = model.predict(df)

        # Add predictions to the dataframe
        df['Predicted_Value'] = predictions


     

        # Display all predictions
        st.subheader('Todas las predicciones')
        st.dataframe(df[['Predicted_Value']])

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

else:
    st.info("Por favor, sube un archivo Excel para comenzar.")
