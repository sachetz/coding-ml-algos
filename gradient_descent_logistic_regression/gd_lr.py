import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

mpl.rc('figure', figsize=[10,6])


def prediction(X, theta):
    return (1/(1 + np.exp(-np.matmul(X, theta))))


def loss(X, y, theta, lamb):
    yhat = prediction(X, theta)
    return (-((y * np.log(yhat)) + 
            ((1 - y) * np.log(1-yhat))
            ).mean(axis=0) + 
            (lamb) * (theta**2).sum(axis=0)
        )


def gradient_descent(X, y, alpha, lamb, T, theta_init=None):
    theta = theta_init.copy()
    m = len(X)
    X_preproc = np.concatenate((np.array([1 for _ in range(m)])[:, np.newaxis], X), axis=1)
    costs = []
    for _ in range(T):
        yhat = prediction(X_preproc, theta)
        theta = theta + alpha * ((1/m) * np.dot(y - yhat, X_preproc) - 2 * lamb * theta)
        costs.append(loss(X_preproc, y, theta, lamb))
    
    plt.plot([i for i in range(T)], costs)
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

    return(theta)


def test_gd():
    X = np.array([[-0.31885351, -1.60298056],
    [-1.53521787, -0.57040089],
    [-0.2167283 ,  0.2548743 ],
    [-0.14944994,  2.01078257],
    [-0.09678416,  0.42220166],
    [-0.22546156, -0.63794309],
    [-0.0162863 ,  1.04421678],
    [-1.08488033, -2.20592483],
    [-0.95121901,  0.83297319],
    [-1.00020817,  0.34346274]])
    y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    alpha = 0.1
    lamb = 1
    theta_init = np.zeros(X.shape[1]+1)
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 1, theta_init),
        np.array([-0.03     ,  0.0189148,  0.0256793]))
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 2, theta_init),
        np.array([-0.05325034,  0.0333282 ,  0.04540004]))
    assert np.allclose(
        gradient_descent(X, y, alpha, lamb, 3, theta_init),
        np.array([-0.07127091,  0.04431147,  0.06054757]))
    print('test_gradient_descent passed')


def calc_theta_and_loss(X_norm, y, alpha, lamb, T, theta_init):
    theta_norm = gradient_descent(X_norm, y, alpha, lamb, T, theta_init)
    print("theta corresponding to normalized inputs:")
    print(theta_norm)
    print()

    X_preproc = np.concatenate((np.array([0 for _ in range(len(X_norm))])[:, np.newaxis], X_norm), axis=1)
    l = loss(X_preproc, y, theta_norm, lamb)
    print(f"Cost using gradient descent's theta: {l}")
    return l, theta_norm


def calc_sklearn_theta_and_loss(X_norm, y, T, lamb):
    clf = LogisticRegression(C = 1 / len(X_norm), max_iter=T, solver="liblinear") #, class_weight={0: 1/len(X), 1: 1/len(X)})
    clf.fit(X_norm, y)
    theta_scikit = np.concatenate((clf.intercept_, clf.coef_.reshape(clf.coef_.shape[1])))
    print("Sklearn's theta values:")
    print(theta_scikit)
    print()

    X_preproc_scikit = np.concatenate((np.array([1 for _ in range(len(X_norm))])[:, np.newaxis], X_norm), axis=1)
    l = loss(X_preproc_scikit, y, theta_scikit, lamb)
    print(f"Cost using sklearn's theta: {l}")
    return l, theta_scikit


def polynomial_features_lr(c1, c2, df, my_color_map):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['mradius', 'mtexture']])
    df[['mradius', 'mtexture']] = pd.DataFrame(scaled_features, columns=['mradius', 'mtexture'])

    poly = PolynomialFeatures(degree=3, include_bias=False)
    logistic_regression_model = make_pipeline(poly, LogisticRegression(max_iter=1000))
    logistic_regression_model.fit(df[[c1, c2]], df['color'])

    plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
    plt.xlabel(c1)
    plt.ylabel(c2)

    x = np.linspace(df[c1].min(), df[c1].max(), 1000)
    y = np.linspace(df[c2].min(), df[c2].max(), 1000)
    xx, yy = np.meshgrid(x,y)
    predicted_prob = logistic_regression_model.predict_proba(
        np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1))
                ))[:,1]
    predicted_prob = predicted_prob.reshape(xx.shape)

    plt.contour(xx, yy, predicted_prob, [0.5], colors=['b'])
    plt.show()


# Data Preprocessing
df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')

X = df.drop(["color", "class", "id"], axis=True)
y = df.loc[:, df.columns == "color"]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
y = y.to_numpy().reshape((y.shape[0]))

theta_init = np.zeros(X_norm.shape[1] + 1)
alpha = 0.1
lamb = 0.5
T = 100

c1 = 'mradius'
c2 = 'mtexture'

test_gd()
print("\n")
cost_local, theta_local = calc_theta_and_loss(X_norm, y, alpha, lamb, T, theta_init)
print("\n")
cost_sklearn, theta_sklearn = calc_sklearn_theta_and_loss(X_norm, y, T, lamb)
print()
cost_diff = (abs(cost_local - cost_sklearn) / max(cost_local, cost_sklearn))
print(f"Cost difference: {cost_diff * 100}%")
max_theta_diff = np.max(np.abs(theta_local - theta_sklearn) / np.maximum(theta_local, theta_sklearn))
print(f"""Max theta difference: {max_theta_diff * 100}%""")
print("\n")
polynomial_features_lr(c1, c2, df, my_color_map)
print()
