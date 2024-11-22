from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model
