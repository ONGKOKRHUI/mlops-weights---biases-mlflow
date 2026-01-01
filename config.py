# Hyperparameter configuration
PARAMS = {
    "epochs": 1,
    "classes": 10,
    "kernels": [16, 32], 
    "batch_size": 128,
    "learning_rate": 0.005,
    "dataset": "MNIST",
    "architecture": "CNN",
    "db_path": "data/mnist.db",  # Path to the SQLite database
    "model_path": "model/mnist.onnx"
}