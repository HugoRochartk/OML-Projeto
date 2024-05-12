""""Modelo de Regressão Logística"""

import numpy as np

from tqdm import tqdm

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, threshold=0.5, multi_class=False):
        self.lr = lr
        self.num_iter = num_iter
        self.multi_class = multi_class
        self.threshold = threshold

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / e_z.sum(axis=1, keepdims=True)

    def _loss(self, h, y):
        if self.multi_class:
            # Using cross entropy loss for multinomial logistic regression
            return -np.mean(np.sum(y * np.log(h), axis=1))
        else:
            # Using binary cross entropy loss for binomial logistic regression
            return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        if self.multi_class:
            k = np.max(y) + 1
            y = np.eye(k)[y]  # Convert labels to one-hot encoding
            self.theta = np.zeros((X.shape[1], k))
        else:
            self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            if self.multi_class:
                h = self._softmax(z)
            else:
                h = self._sigmoid(z)
            
            gradient = np.dot(X.T, (h - y)) / y.shape[0]
            self.theta -= self.lr * gradient
            
            if i % 100 == 0:
                print(f'loss: {self._loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.multi_class:
            return self._softmax(np.dot(X, self.theta))
        return self._sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        if self.multi_class:
            return np.argmax(self.predict_prob(X), axis=1)
        return self.predict_prob(X) >= self.threshold


                
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=2, n_classes=3, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    X_train, y_train, X_test, y_test = X[:80], y[:80], X[80:], y[80:]

    # Example usage:
    # For binomial logistic regression:
    # model = LogisticRegression(lr=0.1, num_iter=3000)
    # model.fit(X_train, y_train)  # y_train should be binary
    # predictions = model.predict(X_test)

    # For multinomial logistic regression:
    model = LogisticRegression(lr=0.1, num_iter=3000, multi_class=True)
    print("Started Training")
    model.fit(X_train, y_train)  # y_train should contain class labels 0,1,2,...,k-1
    print("Finished Training")
    print("Started Predicting")
    predictions = model.predict(X_test)
    print("Finished Predicting")