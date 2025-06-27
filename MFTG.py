import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

st.title('Predicción de Mediciones Futuras')

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
            scaler = joblib.load('/content/standard_scaler.pkl')
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
            model = joblib.load('/content/best_random_forest_regressor_model_gridsearch.pkl')
        except FileNotFoundError:
            st.error("Error: Archivo 'best_random_forest_regressor_model_gridsearch.pkl' no encontrado. Asegúrate de que el archivo esté en '/content/'.")
            st.stop()


        # Perform predictions
        predictions = model.predict(df)

        # Add predictions to the dataframe
        df['Predicted_Value'] = predictions

        st.subheader('Predicciones')
        st.dataframe(df[['Predicted_Value']].head())


        # --- Evaluation (Optional, requires true values) ---
        st.subheader('Evaluación (Requiere valores reales)')

        # Assuming you might have true values in a column, or you might want to add a way to input them
        # For demonstration, let's assume the uploaded data *might* have a 'True_Value' column
        # In a real application predicting future values, you won't have true values for the *future* data.
        # This evaluation section is more relevant if you are using this script to evaluate predictions on historical data with known true values.
        if 'True_Value' in df.columns:
            st.write("Calculando métricas de evaluación...")
            try:
                # Calculate R2 Score
                r2 = r2_score(df['True_Value'], df['Predicted_Value'])
                st.write(f'R2 Score: {r2:.4f}')

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(df['True_Value'], df['Predicted_Value']))
                st.write(f'RMSE: {rmse:.4f}')

                # Calculate MAE
                mae = mean_absolute_error(df['True_Value'], df['Predicted_Value'])
                st.write(f'MAE: {mae:.4f}')

                # Calculate MAPE
                # Handle cases where true value is zero to avoid division by zero
                # Add a small epsilon to avoid division by zero if true values can be 0
                epsilon = 1e-8
                mape = np.mean(np.abs((df['True_Value'] - df['Predicted_Value']) / (df['True_Value'] + epsilon))) * 100
                st.write(f'MAPE: {mape:.4f}%')

            except Exception as e:
                 st.error(f"Error durante el cálculo de las métricas de evaluación. Asegúrate de que las columnas 'True_Value' y 'Predicted_Value' no contienen valores nulos o infinitos, y tienen tipos de datos numéricos. Detalle: {e}")
        else:
            st.info("Para calcular métricas de evaluación (R2, RMSE, MAE, MAPE), el archivo Excel debe contener una columna llamada 'True_Value' con los valores reales.")

        # Display all predictions
        st.subheader('Todas las predicciones')
        st.dataframe(df[['Predicted_Value']])

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

else:
    st.info("Por favor, sube un archivo Excel para comenzar.")
