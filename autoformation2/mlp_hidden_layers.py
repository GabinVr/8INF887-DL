import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import struct
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import expit
    import sys
    return expit, mo, np, os, plt, struct, sys


@app.cell
def _(mo):
    mo.md(r"""
    # 8INF887 - Deep learning
    ## Autoformation 2: *Vous devez tenter de transformer le modèle afin qu'il supporte un nombre arbitraire de couches cachées. Enfin, vous devez le tester sur un autre dataset de votre choix.*
    ### Gabin VRILLAULT - Février 2026
    """)
    return


@app.cell
def _(expit, np, sys):
    class NeuralNetMLP(object):
        """ Feedforward neural network / Multi-layer perceptron classifier.

        Parameters
        ------------
        n_output : int
            Number of output units, should be equal to the number of unique class labels.
        n_features : int
            Number of features (dimensions) in the target dataset.Should be equal to the number of columns in the X array.
        n_hidden : int (default: 30)
            Number of hidden units.
        n_hidden_layers : int 
            Number of hidden layers
        l1 : float (default: 0.0)
            Lambda value for L1-regularization. No regularization if l1=0.0 (default)
        l2 : float (default: 0.0)
            Lambda value for L2-regularization. No regularization if l2=0.0 (default)
        epochs : int (default: 500)
            Number of passes over the training set.
        eta : float (default: 0.001)
            Learning rate.
        alpha : float (default: 0.0)
            Momentum constant. Factor multiplied with the gradient of the previous epoch t-1 to improve learning speed
            w(t) := w(t) - (grad(t) + alpha*grad(t-1))
        decrease_const : float (default: 0.0)
            Decrease constant. Shrinks the learning rate after each epoch via eta / (1 + epoch*decrease_const)
        shuffle : bool (default: True)
            Shuffles training data every epoch if True to prevent circles.
        minibatches : int (default: 1)
            Divides training data into k minibatches for efficiency. Normal gradient descent learning if k=1 (default).
        random_state : int (default: None)
            Set random state for shuffling and initializing the weights.

        Attributes
        -----------
        cost_ : list
          Sum of squared errors after each epoch.

        """

        def __init__(self, n_output, n_features, n_hidden=30, n_hidden_layers=1, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0,
                     decrease_const=0.0, shuffle=True, minibatches=1, random_state=None):

            np.random.seed(random_state)
            self.n_output = n_output
            self.n_features = n_features
            self.n_hidden = n_hidden
            self.n_hidden_layers = n_hidden_layers 
            self.W = self._initialize_weights()
            self.l1 = l1
            self.l2 = l2
            self.epochs = epochs
            self.eta = eta
            self.alpha = alpha
            self.decrease_const = decrease_const
            self.shuffle = shuffle
            self.minibatches = minibatches
        
        def _encode_labels(self, y, k):
            """Encode labels into one-hot representation

            Parameters
            ------------
            y : array, shape = [n_samples]   Target values.

            Returns
            -----------
            onehot : array, shape = (n_labels, n_samples)

            """
            onehot = np.zeros((k, y.shape[0]))
            for idx, val in enumerate(y):
                onehot[val, idx] = 1.0
            return onehot

        def _initialize_weights(self):
            """Initialize weights with small random numbers."""
            W = []
            w1 = np.random.uniform(-1.0, 1.0,
                                   size=self.n_hidden * (self.n_features + 1))
            w1 = w1.reshape(self.n_hidden, self.n_features + 1)
            W.append(w1)
            for i in range(1, self.n_hidden_layers):
                w = np.random.uniform(-1.0, 1.0,
                                      size=self.n_hidden * (self.n_hidden + 1))
                w = w.reshape(self.n_hidden, self.n_hidden + 1)
                W.append(w)
            w2 = np.random.uniform(-1.0, 1.0,
                                   size=self.n_output * (self.n_hidden + 1))
            w2 = w2.reshape(self.n_output, self.n_hidden + 1)
            W.append(w2)
            return W

        def _sigmoid(self, z):
            """Compute logistic function (sigmoid)

            Uses scipy.special.expit to avoid overflow
            error for very small input values z.

            """
            # return 1.0 / (1.0 + np.exp(-z))
            return expit(z)

        def _sigmoid_gradient(self, z):
            """Compute gradient of the logistic function"""
            sg = self._sigmoid(z)
            return sg * (1.0 - sg)

        def _add_bias_unit(self, X, how='column'):
            """Add bias unit (column or row of 1s) to array at index 0"""
            if how == 'column':
                X_new = np.ones((X.shape[0], X.shape[1] + 1))
                X_new[:, 1:] = X
            elif how == 'row':
                X_new = np.ones((X.shape[0] + 1, X.shape[1]))
                X_new[1:, :] = X
            else:
                raise AttributeError('`how` must be `column` or `row`')
            return X_new

        def _feedforward(self, X, W):
            """Compute feedforward step

            Parameters
            -----------
            X : array, shape = [n_samples, n_features]
                Input layer with original features.
            W : list of arrays
                List of weight matrices for each layer.
        
            Returns
            ----------
            A : list of arrays
                List of activation of each layer.
            Z : list of arrays
                List of net input of each layer.
            """
            A = [self._add_bias_unit(X, how='column')]
            Z = []
            for i, w in enumerate(W):
                a_prev = A[-1] # On prend la dernière activation calculée
                if i == 0:
                    z_next = w.dot(a_prev.T)
                else:
                    z_next = w.dot(a_prev)
                a_next = self._sigmoid(z_next)
                if i != len(W) - 1:  
                    a_next = self._add_bias_unit(a_next, how='row')
                A.append(a_next)
                Z.append(z_next)

            return A, Z

        def _L2_reg(self, lambda_, W):
            """Compute L2-regularization cost"""
            return (lambda_ / 2.0) * sum(np.sum(w[:, 1:] ** 2) for w in W)

        def _L1_reg(self, lambda_, W):
            """Compute L1-regularization cost"""
            return (lambda_ / 2.0) * sum(np.abs(w[:, 1:]).sum() for w in W)

        def _get_cost(self, y_enc, output, W):
            """Compute cost function.

            Parameters
            ----------
            y_enc : array, shape = (n_labels, n_samples)
                one-hot encoded class labels.
            output : array, shape = [n_output_units, n_samples]
                Activation of the output layer (feedforward)
            W : list of arrays
                List of weight matrices for each layer.
            Returns
            ---------
            cost : float
                Regularized cost.

            """
            term1 = -y_enc * (np.log(output))
            term2 = (1.0 - y_enc) * np.log(1.0 - output)
            cost = np.sum(term1 - term2)
            L1_term = self._L1_reg(self.l1, W)
            L2_term = self._L2_reg(self.l2, W)
            cost = cost + L1_term + L2_term
            return cost

        #
        # Nous verrons plus tard
        #
        def _get_gradient(self, A, Z, y_enc, W):
            """ Compute gradient step using backpropagation.

            Parameters
            ------------
            A : list of arrays
                List of activation of each layer.
            Z : list of arrays
                List of net input of each layer.
            y_enc : array, shape = (n_labels, n_samples)
                one-hot encoded class labels.
            W : list of arrays
                List of weight matrices for each layer.

            Returns
            ---------
            grads : list of arrays
                List of gradient for each layer.

            """
            grads = [None] * len(W)
            # backpropagation
            sigma = A[-1] - y_enc  # erreur de classification

            for i in range(len(W) - 1, 0, -1): # On itère à l'envers 
                Z_prev = Z[i-1]
            
                Z_prev = self._add_bias_unit(Z_prev, how='row')
                sigma_next = W[i].T.dot(sigma) * self._sigmoid_gradient(Z_prev)
                sigma_next = sigma_next[1:, :]

                grads[i] = sigma.dot(A[i].T)
                sigma = sigma_next
        
            grads[0] = sigma.dot(A[0])
        
            for i,_ in enumerate(grads):
                grads[i][:, 1:] += self.l2 * W[i][:, 1:]
                grads[i][:, 1:] += self.l1 * np.sign(W[i][:, 1:])
            # L1Effect = 0.1 * np.sign(w1[:, 1:])
            # L2Effect = 0.1 * w1[:, 1:]

            return grads

        def predict(self, X):
            """Predict class labels

            Parameters
            -----------
            X : array, shape = [n_samples, n_features]
                Input layer with original features.

            Returns:
            ----------
            y_pred : array, shape = [n_samples]
                Predicted class labels.

            """
            if len(X.shape) != 2:
                raise AttributeError('X must be a [n_samples, n_features] array.\n'
                                     'Use X[:,None] for 1-feature classification,'
                                     '\nor X[[i]] for 1-sample classification')

            A, _ = self._feedforward(X, self.W)
            y_pred = np.argmax(A[-1], axis=0)
            return y_pred

        #
        # Fonction d'entraînement
        #
        def fit(self, X, y, print_progress=False):
            """ Learn weights from training data.

            Parameters
            -----------
            X : array, shape = [n_samples, n_features]
                Input layer with original features.
            y : array, shape = [n_samples]
                Target class labels.
            print_progress : bool (default: False)
                Prints progress as the number of epochs
                to stderr.

            Returns:
            ----------
            self

            """
            self.cost_ = []
            X_data, y_data = X.copy(), y.copy()
            y_enc = self._encode_labels(y, self.n_output)  # Vecteur one-hot

            delta_W_prev = [np.zeros(w.shape) for w in self.W]  

            for i in range(self.epochs):  # Nombre de passage sur le dataset

                # adaptive learning rate
                self.eta /= (1 + self.decrease_const * i)  # Permet de réduire le nombre d'epochs nécessaire à la convergence en limitant les risques de "pas" trop grand!

                if print_progress:
                    sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                    sys.stderr.flush()

                if self.shuffle:  # on mélange le dataset à chaque epoch
                    idx = np.random.permutation(y_data.shape[0])
                    X_data, y_enc = X_data[idx], y_enc[:, idx]
            
                mini = np.array_split(range(y_data.shape[0]),
                                      self.minibatches)  # Si le mode minibatch est activé, le dataset en entrée est divisé en batch pour le calcul des gradients
                for idx in mini:
                    # feedforward
                    A, Z = self._feedforward(X_data[idx], self.W)  # Ce que nous avons vu jusqu'à présent

                    cost = self._get_cost(y_enc=y_enc[:, idx], output=A[-1], W=self.W)
                    self.cost_.append(cost)

                    # compute gradient via backpropagation
                    #
                    # Nous verrons plus en détails
                    grads = self._get_gradient(A=A, Z=Z, y_enc=y_enc[:, idx], W=self.W)


                    delta_w = [self.eta * grad for grad in grads]
                    self.W = [self.W[i] - (delta_w[i] + (self.alpha * delta_W_prev[i])) for i in range(len(self.W))]
                    delta_W_prev = delta_w

            return self


    return (NeuralNetMLP,)


@app.cell
def _(mo):
    mo.md(r"""
    Pour adapter à un nombre quelconque de couches cachées le code fournit du MLP j'ai remplacé les variables de poids $w_1,w_2$ par une liste de poids $W$ , de même pour les unités d'activation et les entrées nettes que j'ai placé dans des listes $A,Z$.
    J'ai également ajouté un paramètre `n_hidden_layers` dans le constructeur pour gérer le nombre de couches cachées.
    ```py
    self.n_hidden_layers = n_hidden_layers
    self.W = self._initialize_weights()
    ```
    J'ai donc modifié la fonction pour initialisé les poids:
    ```py
    def _initialize_weights(self):
        "Initialize weights with small random numbers."
        W = []
        w1 = np.random.uniform(-1.0, 1.0,
                               size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        W.append(w1)
        for i in range(1, self.n_hidden_layers):
            w = np.random.uniform(-1.0, 1.0,
                                  size=self.n_hidden * (self.n_hidden + 1))
            w = w.reshape(self.n_hidden, self.n_hidden + 1)
            W.append(w)
        w2 = np.random.uniform(-1.0, 1.0,
                               size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        W.append(w2)
        return W
    ```

    Puis j'ai modifié la méthode privée `_feedforward()` pour quelle prenne en compte $A,Z$ et $W$
    ```py
    def _feedforward(self, X, W):
        '''Compute feedforward step

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        W : list of arrays
            List of weight matrices for each layer.

        Returns
        ----------
        A : list of arrays
            List of activation of each layer.
        Z : list of arrays
            List of net input of each layer.
        '''
        A = [self._add_bias_unit(X, how='column')]
        Z = []
        for i, w in enumerate(W):
            a_prev = A[-1] # On prend la dernière activation calculée
            if i == 0:
                z_next = w.dot(a_prev.T)
            else:
                z_next = w.dot(a_prev)
            a_next = self._sigmoid(z_next)
            if i != len(W) - 1:
                a_next = self._add_bias_unit(a_next, how='row')
            A.append(a_next)
            Z.append(z_next)

        return A, Z
    ```
    J'ai mis à jour la fonction de coût ainsi que les fonctions de régularisation:
    ```py
    def _L2_reg(self, lambda_, W):
        '''Compute L2-regularization cost'''
        return (lambda_ / 2.0) * sum(np.sum(w[:, 1:] ** 2) for w in W)

    def _L1_reg(self, lambda_, W):
        '''Compute L1-regularization cost'''
        return (lambda_ / 2.0) * sum(np.abs(w[:, 1:]).sum() for w in W)

    def _get_cost(self, y_enc, output, W):
        '''Compute cost function.

        Parameters
        ----------
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        output : array, shape = [n_output_units, n_samples]
            Activation of the output layer (feedforward)
        W : list of arrays
            List of weight matrices for each layer.
        Returns
        ---------
        cost : float
            Regularized cost.

        '''
        term1 = -y_enc * (np.log(output))
        term2 = (1.0 - y_enc) * np.log(1.0 - output)
        cost = np.sum(term1 - term2)
        L1_term = self._L1_reg(self.l1, W)
        L2_term = self._L2_reg(self.l2, W)
        cost = cost + L1_term + L2_term
        return cost
    ```
    Le plus difficile, la méthode `_get_gradient`:
    ```py
    def _get_gradient(self, A, Z, y_enc, W):
        ''' Compute gradient step using backpropagation.

        Parameters
        ------------
        A : list of arrays
            List of activation of each layer.
        Z : list of arrays
            List of net input of each layer.
        y_enc : array, shape = (n_labels, n_samples)
            one-hot encoded class labels.
        W : list of arrays
            List of weight matrices for each layer.

        Returns
        ---------
        grads : list of arrays
            List of gradient for each layer.

        '''
        grads = [None] * len(W)
        # backpropagation
        sigma = A[-1] - y_enc  # erreur de classification

        for i in range(len(W) - 1, 0, -1): # On itère à l'envers
            Z_prev = Z[i-1]

            Z_prev = self._add_bias_unit(Z_prev, how='row')
            sigma_next = W[i].T.dot(sigma) * self._sigmoid_gradient(Z_prev)
            sigma_next = sigma_next[1:, :]

            grads[i] = sigma.dot(A[i].T)
            sigma = sigma_next

        grads[0] = sigma.dot(A[0])

        for i,_ in enumerate(grads):
            grads[i][:, 1:] += self.l2 * W[i][:, 1:]
            grads[i][:, 1:] += self.l1 * np.sign(W[i][:, 1:])
        # L1Effect = 0.1 * np.sign(w1[:, 1:])
        # L2Effect = 0.1 * w1[:, 1:]

        return grads
    ```
    Finalement, la méthode `fit`:
    ```py
    def fit(self, X, y, print_progress=False):
        ''' Learn weights from training data.

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        y : array, shape = [n_samples]
            Target class labels.
        print_progress : bool (default: False)
            Prints progress as the number of epochs
            to stderr.

        Returns:
        ----------
        self

        '''
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._encode_labels(y, self.n_output)  # Vecteur one-hot

        delta_W_prev = [np.zeros(w.shape) for w in self.W]

        for i in range(self.epochs):  # Nombre de passage sur le dataset

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * i)  # Permet de réduire le nombre d'epochs nécessaire à la convergence en limitant les risques de "pas" trop grand!

            if print_progress:
                sys.stderr.write('\rEpoch: %d/%d' % (i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:  # on mélange le dataset à chaque epoch
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]),
                                  self.minibatches)  # Si le mode minibatch est activé, le dataset en entrée est divisé en batch pour le calcul des gradients
            for idx in mini:
                # feedforward
                A, Z = self._feedforward(X_data[idx], self.W)  # Ce que nous avons vu jusqu'à présent

                cost = self._get_cost(y_enc=y_enc[:, idx], output=A[-1], W=self.W)
                self.cost_.append(cost)

                # compute gradient via backpropagation
                #
                # Nous verrons plus en détails
                grads = self._get_gradient(A=A, Z=Z, y_enc=y_enc[:, idx], W=self.W)


                delta_w = [self.eta * grad for grad in grads]
                self.W = [self.W[i] - (delta_w[i] + (self.alpha * delta_W_prev[i])) for i in range(len(self.W))]
                delta_W_prev = delta_w

        return self
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Testons cette implémentation avec `mnist`
    """)
    return


@app.cell
def _(NeuralNetMLP, mo, np, os, plt, struct):
    mo.show_code()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    def load_mnist(path, kind='train'):
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        return images, labels

    path = '../ressources/code/perceptron multi-couche'
    X_train, y_train = load_mnist(path, kind='train')
    X_test, y_test = load_mnist(path, kind='t10k')


    nn = NeuralNetMLP(n_output=10,
                      n_features=X_train.shape[1],
                      n_hidden=50,
                      n_hidden_layers=2,
                      l2=0.1,
                      l1=0.0,
                      epochs=1000,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      minibatches=50,
                      shuffle=True,
                      random_state=1)

    nn.fit(X_train, y_train, print_progress=True)

    plt.plot(range(len(nn.cost_)), nn.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()
    # plt.savefig('./figures/cost.png', dpi=300)
    plt.show()

    batches = np.array_split(range(len(nn.cost_)), 1000)
    cost_ary = np.array(nn.cost_)
    cost_avgs = [np.mean(cost_ary[i]) for i in batches]

    plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.tight_layout()
    #plt.savefig('./figures/cost2.png', dpi=300)
    plt.show()


    y_train_pred = nn.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (acc * 100))

    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (acc * 100))


    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test != y_test_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('./figures/mnist_miscl.png', dpi=300)
    plt.show()


    return


@app.cell
def _(mo):
    mo.image("./public/image.png")
    return


@app.cell
def _(mo):
    mo.image("./public/image2.png")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Training accuracy: 98.02%

    Test accuracy: 96.24%
    """)
    return


@app.cell
def _(mo):
    mo.image("./public/image3.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Maintenant testons avec un autre dataset: `CIFAR-10`
    """)
    return


@app.cell
def _(NeuralNetMLP, mo, plt):
    import pickle

    def unpickle(file): 
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    batch1 = unpickle('./cifar-10-batches-py/data_batch_1')
    X_train_cifar10 = batch1[b'data']
    y_train_cifar10 = batch1[b'labels']
    batch2 = unpickle('./cifar-10-batches-py/data_batch_2')
    X_test_cifar10 = batch2[b'data']
    y_test_cifar10 = batch2[b'labels']
    print(f"X_train_cifar10 shape: {X_train_cifar10.shape}, y_train_cifar10 shape: {len(y_train_cifar10)}")
    print(f"X_test_cifar10 shape: {X_test_cifar10.shape}, y_test_cifar10 shape: {len(y_test_cifar10)}")
    X_train_cifar10 = X_train_cifar10.astype('float32') / 255.0
    X_test_cifar10 = X_test_cifar10.astype('float32') / 255.0

    labels_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    def display():
        img_cifar = X_train_cifar10[0].reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape and transpose to get the correct format for
        plt.imshow(img_cifar)
        plt.title(f"Label: {labels_map[y_train_cifar10[0]]}")
        plt.axis('off')
        plt.show()

    mo.show_code(display())

    nn_cifar = NeuralNetMLP(n_output=10,
                      n_features=X_train_cifar10.shape[1],
                      n_hidden=50,
                      n_hidden_layers=4,
                      l2=0.1,
                      l1=0.0,
                      epochs=500,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      minibatches=50,
                      shuffle=True,
                      random_state=1)   
    return (
        X_test_cifar10,
        X_train_cifar10,
        nn_cifar,
        y_test_cifar10,
        y_train_cifar10,
    )


@app.cell
def _(mo):
    mo.image("public/frog.png")
    return


@app.cell
def _(X_train_cifar10, nn_cifar, np, plt, y_test_cifar10, y_train_cifar10):
    y_train_ = np.array(y_train_cifar10, dtype=int)
    y_test_ = np.array(y_test_cifar10, dtype=int)



    nn_cifar.fit(X_train_cifar10, y_train_, print_progress=True)

    plt.plot(range(len(nn_cifar.cost_)), nn_cifar.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()

    plt.show()



    return y_test_, y_train_


@app.cell
def _(X_test_cifar10, X_train_cifar10, nn_cifar, np, y_test_, y_train_):
    # Let's evaluate the performance of the model on the test sets
    y_train_pred_cifar10 = nn_cifar.predict(X_train_cifar10)
    accuracy_train = np.sum(y_train_ == y_train_pred_cifar10, axis=0) / X_train_cifar10.shape[0]


    y_test_pred_cifar10 = nn_cifar.predict(X_test_cifar10)
    accuracy_test = np.sum(y_test_ == y_test_pred_cifar10, axis=0) / X_test_cifar10.shape[0]

    return accuracy_test, accuracy_train


@app.cell
def _(accuracy_test, accuracy_train, mo):
    mo.md(f"""
    Le modèle a obtenu sur CIFAR-10 la précision de 
    {accuracy_train*100:.2f}% sur l'ensemble d'entrainement
    {accuracy_test*100:.2f}% sur l'ensemble de test

    Ce qui est loin d'être optimal et montre un sur-apprentissage flagrant.
    Il faudrait optimiser le réseau pour avoir de meilluers résultats
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
