# KNN Regressor

The code implements an sklearn-style regressor class KNNRegressor for k-nearest-neighbor linear regression. It generates the data based on a model, plots the data points, the underlying function without the noise $y(x)−\epsilon$, and the hypothesis function for k-nearest-neighbor linear regression.

# Locally Weighted Regressor

The code also implements Locally Weighted Regressor in which the kernel is the rbf kernel that takes a user specified $\gamma$ parameter. It also creates a combined plot that includes
- data points
- the underlying function without the noise
- the hypothesis function for k-nearest-neighbor linear regression with k = 5
- the hypothesis function for locally weighted regression with γ = 1/40.

# Best Values

Then, using crossvalidation on the above dataset, it determines the best value of
- k for k-nearest-neighbor linear regression
- $\gamma$ for locally weighted regression