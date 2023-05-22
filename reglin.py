import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Título de la aplicación
st.title("Regresión Lineal")

# Subtítulo
st.subheader("Ingrese los datos")

# Obtener los datos del usuario
x_values = st.text_input("Ingrese los valores de x separados por coma:")
y_values = st.text_input("Ingrese los valores de y separados por coma:")

# Convertir los datos a listas numéricas
x_values = [float(x) for x in x_values.split(",") if x.strip()]
y_values = [float(y) for y in y_values.split(",") if y.strip()]

# Verificar si hay suficientes datos para realizar la regresión lineal
if len(x_values) > 1 and len(y_values) > 1 and len(x_values) == len(y_values):
    # Crear un DataFrame con los datos
    data = pd.DataFrame({'x': x_values, 'y': y_values})

    # Realizar la regresión lineal
    reg = LinearRegression()
    reg.fit(data[['x']], data['y'])

    # Obtener los coeficientes de la regresión lineal
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Visualizar la línea de regresión y los datos
    fig, ax = plt.subplots()
    ax.scatter(data['x'], data['y'], label='Datos')
    ax.plot(data['x'], reg.predict(data[['x']]), color='red', label='Regresión Lineal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Regresión Lineal')
    ax.legend()
    st.pyplot(fig)

    # Mostrar los coeficientes de la regresión lineal
    st.subheader("Resultados")
    st.write(f"Pendiente (slope): {slope}")
    st.write(f"Intercepto (intercept): {intercept}")
else:
    st.write("Ingrese al menos 2 valores válidos para x e y.")

