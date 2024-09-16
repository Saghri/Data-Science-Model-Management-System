from models.base_model import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DecisionTreeModel(Model):
    def __init__(self, dataset, hyperparameters=None):
        super().__init__('Decision Tree', dataset, hyperparameters)
        self.model = DecisionTreeClassifier(**self._hyperparameters)

    def train(self):
        X_train, y_train = self.dataset['train']
        self.model.fit(X_train, y_train)

    def evaluate(self):
        X_test, y_test = self.dataset['test']
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)
