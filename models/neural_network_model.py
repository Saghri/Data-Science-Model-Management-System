import tensorflow as tf
from models.base_model import Model
from sklearn.metrics import mean_squared_error

class NeuralNetworkModel(Model):
    def __init__(self, dataset, hyperparameters=None):
        super().__init__('Neural Network', dataset, hyperparameters)
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.dataset['train'][0].shape[1],)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))  # Assuming regression problem

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self):
        X_train, y_train = self.dataset['train']
        # Ensure y_train is properly encoded for regression tasks
        if y_train.ndim == 1:  # Check if y_train needs reshaping
            y_train = y_train.reshape(-1, 1)
        
        # Train the model
        self.model.fit(X_train, y_train, epochs=self._hyperparameters.get('epochs', 10), verbose=1)

    def evaluate(self):
        X_test, y_test = self.dataset['test']
        y_pred = self.model.predict(X_test)
        # Calculate Mean Squared Error for regression
        mse = mean_squared_error(y_test, y_pred)
        return mse
