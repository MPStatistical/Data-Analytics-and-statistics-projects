import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, kstest
from statsmodels.stats.diagnostic import het_white

# Configuración de estilo
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12})

# Título de la aplicación
st.title("Regresión Lineal")

# Subtítulo
st.subheader("Ingrese los datos")

# Obtener los datos del usuario
x_values = st.text_input("Ingrese los valores de x separados por coma: (Ej: 0.0, 0.20, 0.41, 0.61)")
y_values = st.text_input("Ingrese los valores de y separados por coma: (Ej: -0.61, 0.40, 1.16, 1.81) ")

# Convertir los datos a listas numéricas
x_values = [float(x) for x in x_values.split(",") if x.strip()]
y_values = [float(y) for y in y_values.split(",") if y.strip()]

# Verificar si hay suficientes datos para realizar la regresión lineal
if len(x_values) > 1 and len(y_values) > 1 and len(x_values) == len(y_values):
    # Crear un DataFrame con los datos
    data = pd.DataFrame({'x': x_values, 'y': y_values})

    # Realizar la regresión lineal
    results = smf.ols('y ~ x', data=data).fit()

    # Obtener los valores ajustados y los intervalos de predicción
    pred_vals = results.get_prediction().summary_frame()
    lower_pred = pred_vals['obs_ci_lower']
    upper_pred = pred_vals['obs_ci_upper']

    # Crear los subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Visualizar la línea de regresión, los datos y los intervalos de predicción
    axs[0].scatter(data['x'], data['y'], label='Datos', color='skyblue', edgecolor='gray', alpha=0.8)
    axs[0].plot(np.array(data['x']), np.array(results.fittedvalues), color='red', label='Regresión Lineal')
    axs[0].plot(np.array(data['x']), np.array(lower_pred), color='gray', linestyle='--')
    axs[0].plot(np.array(data['x']), np.array(upper_pred), color='gray', linestyle='--')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Regresión Lineal')
    axs[0].legend(loc='upper left')

    # Visualizar los residuos con un QQ-Plot
    sm.qqplot(results.resid, line='s', ax=axs[1], color='gray')
    axs[1].set_title('QQ-Plot')

    # Visualizar los residuos estandarizados frente a los valores ajustados
    residuals_standardized = results.resid_pearson
    axs[2].scatter(results.fittedvalues, residuals_standardized, color='blue', edgecolor='gray', alpha=0.8)
    axs[2].axhline(0, color='gray', linestyle='--')  # Agregar línea horizontal
    axs[2].set_xlabel('Valores ajustados')
    axs[2].set_ylabel('Residuos estandarizados')
    axs[2].set_title('Residuos estandarizados vs Valores ajustados')

    # Ajustar automáticamente el layout
    plt.tight_layout()

    # Mostrar los gráficos
    st.pyplot(fig)

    # Verificar si los residuos siguen una distribución normal
    residuals = results.resid
    if len(residuals) < 50:
        _, p_normal = shapiro(residuals)
        test_used = "Shapiro-Wilk"
    else:
        _, p_normal = kstest(residuals, 'norm')
        test_used = "Kolmogorov-Smirnov"

    # Prueba de homocedasticidad (Prueba de White)
    _, p_white, _, _ = het_white(residuals, results.model.exog)

    # Mostrar el summary del modelo
    st.write(results.summary())

    # Mostrar los resultados de las pruebas
    st.subheader("Resultados de las pruebas")

    if p_normal > 0.05:
        st.write(f'Prueba de {test_used}: Los residuos siguen una distribución normal (p={p_normal:.3f})')
    else:
        st.write(f'Prueba de {test_used}: Los residuos no siguen una distribución normal (p={p_normal:.3f})')

    if p_white > 0.05:
        st.write(f'Prueba de White: Los errores son homocedásticos (p={p_white:.3f})')
    else:
        st.write(f'Prueba de White: Los errores no son homocedásticos (p={p_white:.3f})')
else:
    st.write("Ingrese al menos 2 valores válidos para x e y.")








