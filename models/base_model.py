from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, model_name, dataset, hyperparameters=None):
        self.model_name = model_name
        self.dataset = dataset
        self._hyperparameters = hyperparameters or {}
        self.__model_weights = None  # Private attribute for sensitive information

    def set_hyperparameters(self, params):
        self._hyperparameters.update(params)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
