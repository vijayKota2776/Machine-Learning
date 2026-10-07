import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==============================================================================
# Polynomial Regression Curve Fitting Challenge
# ==============================================================================

# 1. Generate a dataset with a clearly non-linear relationship
# We will use a small sample size to easily visualize the effects of overfitting.
np.random.seed(42)
n_samples = 15
X = np.sort(np.random.rand(n_samples, 1) * 10, axis=0)
# True underlying function is a gentle sine wave
y_true = np.sin(X).ravel()
# Add random noise to simulate real-world data variance
y = y_true + np.random.randn(n_samples) * 0.4

# 2. Fit Polynomial Regression models (Degree 2 and Degree 4)
degrees = [2, 4]
colors = ['blue', 'red']
models = {}

# Generate fine-grained X values for smooth plotting of the curve
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Training Data (with noise)', zorder=5)

for degree, color in zip(degrees, colors):
    # Transform features to polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict on the training data to calculate error
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    
    # Predict on the fine-grained X_plot for visualization
    X_plot_poly = poly_features.transform(X_plot)
    y_plot_pred = model.predict(X_plot_poly)
    
    # Plot the curve
    plt.plot(X_plot, y_plot_pred, color=color, label=f'Degree {degree} (MSE: {mse:.2f})', linewidth=2)
    models[degree] = model

plt.title('Polynomial Regression: Degree 2 vs. Degree 4')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
output_path = 'polynomial_comparison.png'
plt.savefig(output_path)
print(f"Visualization saved to {output_path}")

# ==============================================================================
# Explanation of Overfitting
# ==============================================================================
explanation = """
Explanation: Overfitting in Polynomial Regression
-------------------------------------------------
In this challenge, we compare a Degree 2 polynomial (quadratic) with a Degree 4 polynomial.

1. Degree 2 (Blue Line): 
   This model captures the general downward and upward trend of the data but might be slightly 
   too simple (underfitting) to capture the true underlying sine wave perfectly. However, it 
   is smooth and generalizes reasonably well to the overall shape.

2. Degree 4 (Red Line):
   The Degree 4 model has more flexibility (more parameters). As seen in the plot, it tries much 
   harder to pass exactly through the scattered training points. This results in a curve that 
   wiggles sharply (especially at the edges or between points). 
   
   While the Mean Squared Error (MSE) on the training data is typically lower for the Degree 4 
   model, this is a classic sign of **overfitting**. The model is capturing the random noise 
   in the training set rather than the true underlying pattern. If we introduced new, unseen 
   test data, the Degree 4 model would likely produce massive errors, whereas the smoother 
   Degree 2 model would generalize better.
"""

print(explanation)
# Save explanation to a markdown file
with open('explanation.md', 'w') as f:
    f.write(explanation)
