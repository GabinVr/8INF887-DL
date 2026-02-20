import marimo

__generated_with = "0.19.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.utils import shuffle
    return np, pd, shuffle


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 1. Améliorer le code du Perceptron ou d'Adaline afin qu'il supporte naturellement les problèmes multi-classes
    """)
    return


@app.cell
def _(np):
    class Perceptron(object):
        """Perceptron classifier.

        Parameters
        ------------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.

        Attributes
        -----------
        w_ : 1d-array
            Weights after fitting.
        errors_ : list
            Number of misclassifications (updates) in each epoch.

        """
        def __init__(self, eta=0.01, n_iter=10):
            self.eta = eta
            self.n_iter = n_iter
            self.w_ = None

        def fit(self, X, y):
            """Fit training data.

            Parameters
            ----------
            X : {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
            y : array-like, shape = [n_samples]
                Target values.

            Returns
            -------
            self : object

            """
            self.w_ = np.zeros(1 + X.shape[1])
            self.errors_ = []

            #print("Poids:", self.w_)

            for _ in range(self.n_iter):
                errors = 0
                for xi, target in zip(X, y):
                    # print("Modèle actuel: ", self.w_)
                    # print("Fleur: ",xi," Type: ",np.where(target == 1, "Iris-setosa", "Autre"))
                    update = self.eta * (target - self.predict(xi))
                    # print("Z=", self.w_[0], "*1+", self.w_[1:], "*", xi, " = ", np.dot(xi, self.w_[1:]) + self.w_[0])
                    # print("Update = ", self.eta, " * (",target," - ", self.predict(xi), ") = ", update)
                    # print("Poids MàJ: ", self.w_[1:], " + ", xi, " * ", update, " = ", end = ' ')
                    self.w_[1:] += update * xi
                    # print(self.w_[1:])

                    self.w_[0] += update
                    errors += int(update != 0.0)
                self.errors_.append(errors)
                # print("ERREURS: ", errors)
            return self

        def net_input(self, X):
            """Calculate net input"""
            return np.dot(X, self.w_[1:]) + self.w_[0]

        def predict(self, X):
            """Return class label after unit step"""
            return np.where(self.net_input(X) >= 0.0, 1, -1)
    return (Perceptron,)


@app.cell
def _(Perceptron, np):
    class PerceptronMultiClass:
        """
        Implementation of One VS All strategie to implement Multiclass classification.
        """
        def __init__(self, eta=0.01, n_iter=10):
            self.perceptrons = []
            self.label_map = {}
            self.eta = eta
            self.n_iter = n_iter


        def fit(self, X, y):
            classes = np.unique(y)
            for i in range(len(classes)):
                self.label_map[i]=classes[i]
                self.perceptrons.append(Perceptron(self.eta,self.n_iter))

            for perceptron, cls in enumerate(classes):
                y_train_perceptron_i = np.where(y == cls, 1, -1)
                self.perceptrons[perceptron].fit(X, y_train_perceptron_i)

        def predict(self, X):
            results = np.array([perceptron.net_input(X) for perceptron in self.perceptrons])
            prediction = np.argmax(results)
            return self.label_map[prediction]    
    return (PerceptronMultiClass,)


@app.cell
def _(mo):
    mo.md(r"""
    ```py
    class PerceptronMultiClass:
        "Implementation of One VS All strategie to implement Multiclass classification."
        def __init__(self, eta=0.01, n_iter=10):
            self.perceptrons = []
            self.label_map = {}
            self.eta = eta
            self.n_iter = n_iter


        def fit(self, X, y):
            classes = np.unique(y)
            for i in range(len(classes)):
                self.label_map[i]=classes[i]
                self.perceptrons.append(Perceptron(self.eta,self.n_iter))

            for perceptron, cls in enumerate(classes):
                y_train_perceptron_i = np.where(y == cls, 1, -1)
                self.perceptrons[perceptron].fit(X, y_train_perceptron_i)

        def predict(self, X):
            results = np.array([perceptron.net_input(X) for perceptron in self.perceptrons])
            prediction = np.argmax(results)
            return self.label_map[prediction]
    ```

     J'ai choisis de ré-utiliser le code du perceptron qui était donné sur moodle et de faire une nouvelle classe pour implémenter la gestion multiclasses de la classification.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2-Tester avec les Iris et comparez avec une implémentation Scikit-Learn (attention Adaline est simplement une régression linéaire dans Scikit)
    """)
    return


@app.cell
def _(PerceptronMultiClass, pd, shuffle):
    a = PerceptronMultiClass()
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df = shuffle(df, random_state=42)
    X_train = df.iloc[0:10, :4].values
    y_train = df.iloc[0:10, 4].values
    a.fit(X_train, y_train)
    X_test = df.iloc[11, :4].values
    y_test = df.iloc[11, 4]
    result = a.predict(X_test)
    return X_test, X_train, result, y_test, y_train


@app.cell
def _(mo, result, y_test):
    mo.md(f"""
    Mon perceptron multiclasses a predit {result} et la bonne réponse était {y_test}
    """)
    return


@app.cell
def _(X_test, X_train, y_train):
    from sklearn.linear_model import Perceptron as Sk_perceptron

    sk = Sk_perceptron()
    sk.fit(X_train, y_train)
    print(X_test.reshape(1,-1))
    result_sk = sk.predict(X_test.reshape(1,-1))
    return Sk_perceptron, result_sk


@app.cell
def _(mo, result_sk, y_test):
    mo.md(f"""
    Le perceptron de scikit-learn a prédit {result_sk[0]} et le bon résultat était {y_test}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Au final, les deux implémentations ont le même résultat.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3-Tester le tout avec un autre ensemble de UCI de votre choix
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    J'ai choisi de tester sur le dataset 'National poll on healthy aging npha' qui permet de prédire le nombre de docteur qu'un patient a  consulté
    """)
    return


@app.cell
def _(pd):
    national_poll_on_healthy_aging_npha = pd.read_csv('https://archive.ics.uci.edu/static/public/936/data.csv',header=None)

    national_poll_on_healthy_aging_npha.head()
    return (national_poll_on_healthy_aging_npha,)


@app.cell
def _(national_poll_on_healthy_aging_npha):
    from sklearn.model_selection import train_test_split
    data = national_poll_on_healthy_aging_npha
    headers = data.iloc[0,:]
    data = data.iloc[1:,:]
    data = data.sample(frac=1, random_state=42)
    X, y = data.iloc[:,1:], data.iloc[:,0]
    X_train_health, X_test_health, y_train_health, y_test_health = train_test_split(X,y, stratify=y, test_size=0.5, random_state=42)
    X_train_health = X_train_health.values.astype(float)
    X_test_health = X_test_health.values.astype(float)
    y_train_health = y_train_health.values.astype(float)
    y_test_health = y_test_health.values.astype(float)
    return X_test_health, X_train_health, y_test_health, y_train_health


@app.cell
def _(mo, y_test_health):
    mo.md(f"""
    J'ai séparé mon dataset en deux parties 50/50 une partie de test et une partie d'entrainement de {len(y_test_health)} éléments
    """)
    return


@app.cell
def _(
    PerceptronMultiClass,
    Sk_perceptron,
    X_test_health,
    X_train_health,
    np,
    y_test_health,
    y_train_health,
):
    my_multiclass_perceptron = PerceptronMultiClass()
    sk_perceptron = Sk_perceptron()


    my_multiclass_perceptron.fit(X_train_health, y_train_health)


    results_my_percepton = np.array([ my_multiclass_perceptron.predict(X) for X in X_test_health])

    print(f"My perceptron got: {np.sum(results_my_percepton==y_test_health)/len(y_test_health):.2f}%")

    sk_perceptron.fit(X_train_health, y_train_health)
    scores = sk_perceptron.score(X=X_test_health, y=y_test_health)
    print(f"Sklearn implementation fot {scores:.2f}%")
    return results_my_percepton, scores


@app.cell
def _(mo, np, results_my_percepton, scores, y_test_health):
    mo.md(f"""
    Mon perceptron a eu un score de {np.sum(results_my_percepton==y_test_health)/len(y_test_health)*100:.2f}% 
    tandis que le perceptron de scikit-learn a eu un score de {scores*100:.2f}%
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On peut expliquer cette différence de précision par le fait que Sklearn utilise la stratégie One vs One au lieu de One vs All qui permet d'être plus précis.
    """)
    return


if __name__ == "__main__":
    app.run()
