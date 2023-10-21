import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import shapiro, kstest
from statsmodels.stats.diagnostic import het_white

# Style configuration
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12})

# Application title
st.title("Linear Regression")

# Subtitle
st.subheader("Enter Data")

# Get user input data
x_values = st.text_input("Enter x values separated by comma: (e.g., 0.0, 0.20, 0.41, 0.61)")
y_values = st.text_input("Enter y values separated by comma: (e.g., -0.61, 0.40, 1.16, 1.81) ")

# Convert data to numerical lists
x_values = [float(x) for x in x_values.split(",") if x.strip()]
y_values = [float(y) for y in y_values.split(",") if y.strip()]

# Check if there are enough data points for linear regression
if len(x_values) > 1 and len(y_values) > 1 and len(x_values) == len(y_values):
    # Create a DataFrame with the data
    data = pd.DataFrame({'x': x_values, 'y': y_values})

    # Perform linear regression
    results = smf.ols('y ~ x', data=data).fit()

    # Get adjusted values and prediction intervals
    pred_vals = results.get_prediction().summary_frame()
    lower_pred = pred_vals['obs_ci_lower']
    upper_pred = pred_vals['obs_ci_upper']

    # Create subplots
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Visualize regression line, data, and prediction intervals
    axs[0].scatter(data['x'], data['y'], label='Data', color='skyblue', edgecolor='gray', alpha=0.8)
    axs[0].plot(np.array(data['x']), np.array(results.fittedvalues), color='red', label='Linear Regression')
    axs[0].plot(np.array(data['x']), np.array(lower_pred), color='gray', linestyle='--')
    axs[0].plot(np.array(data['x']), np.array(upper_pred), color='gray', linestyle='--')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Linear Regression')
    axs[0].legend(loc='upper left')

    # Visualize residuals with a QQ-Plot
    sm.qqplot(results.resid, line='s', ax=axs[1], color='gray')
    axs[1].set_title('QQ-Plot')

    # Visualize standardized residuals vs. fitted values
    residuals_standardized = results.resid_pearson
    axs[2].scatter(results.fittedvalues, residuals_standardized, color='blue', edgecolor='gray', alpha=0.8)
    axs[2].axhline(0, color='gray', linestyle='--')
    axs[2].set_xlabel('Fitted Values')
    axs[2].set_ylabel('Standardized Residuals')
    axs[2].set_title('Standardized Residuals vs Fitted Values')

    # Automatically adjust layout
    plt.tight_layout()

    # Show plots
    st.pyplot(fig)

    # Check if residuals follow a normal distribution
    residuals = results.resid
    if len(residuals) < 50:
        _, p_normal = shapiro(residuals)
        test_used = "Shapiro-Wilk"
    else:
        _, p_normal = kstest(residuals, 'norm')
        test_used = "Kolmogorov-Smirnov"

    # Homoscedasticity test (White test)
    _, p_white, _, _ = het_white(residuals, results.model.exog)

    # Show the model summary
    st.write(results.summary())

    # Show test results
    st.subheader("Test Results")

    if p_normal > 0.05:
        st.write(f'{test_used} Test: Residuals follow a normal distribution (p={p_normal:.3f})')
    else:
        st.write(f'{test_used} Test: Residuals do not follow a normal distribution (p={p_normal:.3f})')

    if p_white > 0.05:
        st.write(f'White Test: Errors are homoscedastic (p={p_white:.3f})')
    else:
        st.write(f'White Test: Errors are not homoscedastic (p={p_white:.3f})')
else:
    st.write("Please enter at least 2 valid values for x and y.")









