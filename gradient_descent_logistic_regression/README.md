# Gradient Descent

The function gradient descent(X, y, alpha, lamb, T, theta init) takes an input numpy matrix X of shape (m,n), a output vector y of shape (m,), a scalar learning rate alpha, a regularization strength parameter lamb, the number of iterations T, inital parameter vector theta init, and returns a vector theta of shape (n + 1,). theta is the logistic regression parameter vector theta found by executing the batch gradient descent algorithm for T iterations on the given inputs.

The function also plots the value of the cost (loss + complexity) function vs the iteration number.

# Logistic Regression

The above implemenation fits a logistic regression model on the accompanying breast cancer data, with 30 input variables, and two classes.

The code also fits the sklearn logistic regression model on the dataset, and compares the final loss and theta values.

# Polynomial Features

A logistic regression model is fit on the just the features mradius and mtexture using scikit-learn’s LogisticRegresion, but includes terms up to degree 3—such as mradius^3 and mradius x mtexture^2. The resultant scatter plot contains the malignant cases in red, the benign cases in green, and the decision boundary corresponding to the model in blue.