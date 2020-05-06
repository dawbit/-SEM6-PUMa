import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def polynomial_regression(attributes, deg, results_set):
    regression = PolynomialFeatures(degree=deg)
    polynomial = regression.fit_transform(attributes)
    pol_regression = LinearRegression()
    pol_regression.fit(polynomial, results_set)
    return regression, pol_regression


def sort_two_lists(list1, list2):
    marge = np.concatenate((list1, list2), axis=1)
    sorted_list = sorted(marge, key=lambda x: x[0])
    arguments, values = [], []
    for i in range(0, len(list1)):
        arguments.append(sorted_list[i][0])
        values.append(sorted_list[i][1])
    return arguments, values


def regression_models(attributes, values, number):
    X_train, X_test, y_train, y_test = train_test_split(attributes, values, test_size=0.3)
    X_for_plots = np.linspace(0, 10, 70).reshape(70, 1)
    list_of_polynomials, parameters, train_error, test_error = [], [], [], []
    for i in range(1, number):
        poly_model, linear_model = polynomial_regression(X_train, i, y_train)
        list_of_polynomials.append(linear_model)
        parameters.append(linear_model.coef_)
        y_train_prediction = linear_model.predict(poly_model.fit_transform(X_train))
        y_test_prediction = linear_model.predict(poly_model.fit_transform(X_test))
        train_error.append(mean_squared_error(y_train_prediction, y_train))
        test_error.append(mean_squared_error(y_test_prediction, y_test))
        whole_set_prediction = linear_model.predict(poly_model.fit_transform(attributes))
        args_plot, results_plot = sort_two_lists(attributes, whole_set_prediction)
        plt.subplot(2, 5, i)
        plt.scatter(attributes, values, color='red')
        plt.plot(args_plot, results_plot, "-o", liewidth=2, color='blue')
    plt.show()
    return list_of_polynomials, parameters, train_error, test_error, X_for_plots, y_train_prediction


if __name__ == '__main__':
    np.random.seed(seed=10)
    m = 100
    X = 10 * np.random.randn(m, 1)
    y = 0.5 * X * np.sin(X) + np.random.randn(m, 1)
    plt.scatter(X, y, color='red')
    plt.show()

    arguments, values = sort_two_lists(X, y)
    arguments = np.array(arguments)
    values = np.array(values)
    regression_models(arguments, values, m)
