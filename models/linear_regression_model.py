from models.base_model import Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegressionModel(Model):
    def __init__(self, dataset, hyperparameters=None):
        super().__init__('Linear Regression', dataset, hyperparameters)
        self.model = LinearRegression(**self._hyperparameters)

    def train(self):
        X_train, y_train = self.dataset['train']
        self.model.fit(X_train, y_train)

    def evaluate(self):
        X_test, y_test = self.dataset['test']
        predictions = self.model.predict(X_test)
        return mean_squared_error(y_test, predictions)

def load_california_dataset():
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {'train': (X_train, y_train), 'test': (X_test, y_test)}
