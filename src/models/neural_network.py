import numpy as np

class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        activation="relu",
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        dropout_rate=0.0,
        random_state=None,
    ):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.dropout_rate = float(dropout_rate)
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.weights = []
        self.biases = []
        self._init_parameters()

        self.history = {"loss": [], "accuracy": []}

    def _init_parameters(self):
        num_layers = len(self.layer_sizes)
        for l in range(num_layers - 1):
            n_in = self.layer_sizes[l]
            n_out = self.layer_sizes[l + 1]

            if self.activation == "relu":
                std = np.sqrt(2.0 / n_in)
            else:
                std = np.sqrt(1.0 / n_in)

            W = np.random.randn(n_in, n_out) * std
            b = np.zeros((1, n_out), dtype=np.float64)

            self.weights.append(W)
            self.biases.append(b)

    def _activation(self, Z):
        if self.activation == "relu":
            return np.maximum(0.0, Z)
        elif self.activation == "sigmoid":
            Z_clipped = np.clip(Z, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-Z_clipped))
        else:
            raise ValueError(f"Activation '{self.activation}' khong ho tro.")

    def _activation_derivative(self, Z):
        if self.activation == "relu":
            return (Z > 0.0).astype(float)
        elif self.activation == "sigmoid":
            sig = self._activation(Z)
            return sig * (1.0 - sig)

    def _softmax(self, Z):
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _loss(self, y_true_one_hot, y_pred_prob):
        n_samples = y_true_one_hot.shape[0]
        eps = 1e-15
        prob_clipped = np.clip(y_pred_prob, eps, 1.0 - eps)
        return -np.sum(y_true_one_hot * np.log(prob_clipped)) / n_samples

    def _dropout(self, A, training=True):
        if not training or self.dropout_rate <= 0.0:
            return A, None

        keep_prob = 1.0 - self.dropout_rate
        mask = (np.random.rand(*A.shape) < keep_prob).astype(float)
        A_dropped = (A * mask) / keep_prob
        return A_dropped, mask

    def _forward(self, X, training=True):
        cache = {"Z": [], "A": [X], "masks": []}
        A_current = X
        num_layers = len(self.weights)

        for l in range(num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]

            Z = np.dot(A_current, W) + b
            A = self._activation(Z)

            if training and self.dropout_rate > 0.0:
                A, mask = self._dropout(A, training=True)
                cache["masks"].append(mask)

            cache["Z"].append(Z)
            cache["A"].append(A)
            A_current = A

        W_last = self.weights[-1]
        b_last = self.biases[-1]

        Z_out = np.dot(A_current, W_last) + b_last
        A_out = self._softmax(Z_out)

        cache["Z"].append(Z_out)
        cache["A"].append(A_out)

        return A_out, cache

    def _backward(self, y_true_one_hot, cache):
        gradients = {"dW": [], "db": []}
        num_layers = len(self.weights)
        n_samples = y_true_one_hot.shape[0]

        e = cache["A"][-1] - y_true_one_hot

        for l in reversed(range(num_layers)):
            A_prev = cache["A"][l]

            dW = np.dot(A_prev.T, e) / n_samples
            db = np.sum(e, axis=0, keepdims=True) / n_samples

            gradients["dW"].insert(0, dW)
            gradients["db"].insert(0, db)

            if l > 0:
                W_current = self.weights[l]
                Z_prev = cache["Z"][l - 1]

                e = np.dot(e, W_current.T) * self._activation_derivative(Z_prev)

                if self.dropout_rate > 0.0:
                    mask = cache["masks"][l - 1]
                    keep_prob = 1.0 - self.dropout_rate
                    e = (e * mask) / keep_prob

        return gradients

    def _update_parameters(self, gradients):
        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * gradients["dW"][l]
            self.biases[l] -= self.learning_rate * gradients["db"][l]

    def _one_hot_encode(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1.0
        return one_hot

    def fit(self, X, y):
        num_classes = self.layer_sizes[-1]
        y_one_hot = self._one_hot_encode(y, num_classes)
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                _, cache = self._forward(X_batch, training=True)
                gradients = self._backward(y_batch, cache)
                self._update_parameters(gradients)

            y_pred_prob, _ = self._forward(X, training=False)
            current_loss = self._loss(y_one_hot, y_pred_prob)
            current_acc = self.score(X, y)

            self.history["loss"].append(current_loss)
            self.history["accuracy"].append(current_acc)

        return self

    def predict_proba(self, X):
        probabilities, _ = self._forward(X, training=False)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
        