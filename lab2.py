import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = 3 * np.random.rand(400, 1)
y = 2 + 4 * X + np.random.rand(400, 1)

X_train = X[:-100]
X_test = X[-100:]
y_train = y[:-100]
y_test = y[-100:]

regression = LinearRegression()
regression.fit(X_train, y_train)
prediction = regression.predict(X_test)

print("Współczynnik : ", regression.intercept_, regression.coef_)
print("Błąd średniokwadratowy: ", mean_squared_error(y_test, prediction))
print("r2: ", r2_score(y_test, prediction))

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, prediction, color='blue', linewidth=2)
plt.show()

# SGD
X_train_new = np.c_[np.ones((300, 1)), X_train]


def change_learning_rate(t, t0=5, t1=50):
    return t0 / (t + t1)


def cost_function(X_sample, y_sample, theta):
    hyp = X_sample.dot(theta)
    cost = ((hyp - y_sample).T.dot(hyp - y_sample)) / 2
    return cost[0, 0]


def stochastic_gradnient_descent(X_train, y_train, X_test, y_test, epochs):
    theta = np.random.rand(2, 1)
    error_values = []
    length = len(X_train)
    for epoch in range(epochs):
        for i in range(length):
            index = np.random.randint(length)
            x_sample = X_train[index:index + 1]
            y_sample = y_train[index:index + 1]
            gradient = 2 * x_sample.T.dot(x_sample.dot(theta) - y_sample)
            eta = change_learning_rate(epoch * length + i)
            theta = theta - eta * gradient
            error = cost_function(x_sample, y_sample, theta)
            print(error)
            error_values.append(error)

    sgd_prediction = theta[0] + theta[1] * X_test
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, sgd_prediction, color='blue')
    plt.show()

    print("Błąd średniokwadratowy: ", mean_squared_error(y_test, sgd_prediction))
    print("Współczynnik : ", theta[0], theta[1])

    return sgd_prediction, error_values


prediction, error = stochastic_gradnient_descent(X_train_new, y_train, X_test, y_test, 80)
