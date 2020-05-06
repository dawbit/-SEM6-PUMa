import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=.3)

    dtc = tree.DecisionTreeClassifier(random_state=0)
    dtc.fit(X_train, y_train)
    fig = plt.figure()
    ax = tree.plot_tree(dtc)
    plt.savefig("Iris_Dawid_Bitner.png")

    train_score = dtc.score(X_train, y_train)
    print("wyniki dla zbioru treningowego: ", train_score)
    test_score = dtc.score(X_test, y_test)
    print("wyniki dla zbioru testowego: ", test_score)

    # Skuteczność dla wytrenowanego drzewa wynosi ~95% i 100% skuteczności dla danych treningowych.
    # Współczynnik gini był wiekszy od 0.0 na pięciu gałązkach (w tym na jednej był bardzo blisku zeru - 0.149),
    # dla żadnego z przypadku nie są to wartości znajdujące się na liściahc drzewa (jego końcach).
    # Jako że na liściach gini wynosi zawsze 0.0 - poznacza to stuprocentową  pewność na danych treningowych.
