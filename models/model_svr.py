from sklearn.svm import SVR

class SVRModel:
    def __init__(self, **kwargs):
        self.model = SVR(**kwargs)

    def fit(self, X, y):
        """Entrena el modelo SVR con los datos de entrada X e Y."""
        self.model.fit(X, y)

    def predict(self, X):
        """Genera predicciones para los datos de entrada X."""
        return self.model.predict(X)

    def get_model(self):
        """Devuelve el modelo entrenado."""
        return self.model
