import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from models.linear_regression_model import LinearRegressionModel, load_california_dataset
from models.decision_tree_model import DecisionTreeModel
from models.neural_network_model import NeuralNetworkModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Management System")

        # Header
        self.header = tk.Label(self.root, text="Select a Model to Train and Evaluate", font=("Arial", 18))
        self.header.pack(pady=20)

        # Buttons for different models
        self.lr_button = tk.Button(self.root, text="Linear Regression", command=self.train_linear_regression)
        self.lr_button.pack(pady=10)

        self.dt_button = tk.Button(self.root, text="Decision Tree", command=self.train_decision_tree)
        self.dt_button.pack(pady=10)

        self.nn_button = tk.Button(self.root, text="Neural Network", command=self.train_neural_network)
        self.nn_button.pack(pady=10)

        # Button to load custom dataset
        self.load_custom_button = tk.Button(self.root, text="Load Custom Dataset", command=self.load_custom_dataset)
        self.load_custom_button.pack(pady=10)

        self.custom_dataset = None

    def train_linear_regression(self):
        dataset = load_california_dataset() if not self.custom_dataset else self.custom_dataset
        model = LinearRegressionModel(dataset)
        model.train()
        mse = model.evaluate()
        messagebox.showinfo("Model Evaluation", f"Linear Regression MSE: {mse}")

    def train_decision_tree(self):
        dataset = self.load_iris_dataset() if not self.custom_dataset else self.custom_dataset
        model = DecisionTreeModel(dataset)
        model.train()
        accuracy = model.evaluate()
        messagebox.showinfo("Model Evaluation", f"Decision Tree Accuracy: {accuracy}")

    def train_neural_network(self):
        dataset = self.load_iris_dataset() if not self.custom_dataset else self.custom_dataset
        model = NeuralNetworkModel(dataset)
        model.train()
        evaluation = model.evaluate()
        messagebox.showinfo("Model Evaluation", f"Neural Network Evaluation: {evaluation}")

    def load_custom_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.custom_dataset = self.load_dataset(file_path)
            messagebox.showinfo("Dataset Loaded", "Custom dataset loaded successfully.")

    def load_dataset(self, file_path):
        data = pd.read_csv(file_path)
        X = data.drop('target', axis=1)  # Update 'target' to match your dataset's target column
        y = data['target']  # Update 'target' to match your dataset's target column

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return {'train': (X_train, y_train), 'test': (X_test, y_test)}

    def load_iris_dataset(self):
        data = load_iris()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return {'train': (X_train, y_train), 'test': (X_test, y_test)}

# Running the GUI
if __name__ == "__main__":
    root = tk.Tk()
    gui = ModelGUI(root)
    root.mainloop()
