import wandb
from main import model_pipeline

# 1. Define the Sweep Configuration
sweep_config = {
    'method': 'bayes',  # 'random', 'grid', or 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {'value': 3},
        'classes': {'value': 10},
        'dataset': {'value': "MNIST"},
        'architecture': {'value': "CNN"},
        'db_path': {'value': "data/mnist.db"},
        'model_path': {'value': "model/mnist.onnx"},
        
        # Parameters to tune
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': { 
            'values': [32, 64, 128, 256]
        },
        'kernels': {
            'values': [[16, 32], [32, 64], [16, 64]]
        }
    }
}

def sweep_train():
    # Initialize the run
    # wandb.init() will be called automatically by the agent with params from sweep_config
    wandb.init(project="pytorch-sqlite-sweeps")
    
    # Call your pipeline
    model_pipeline()

if __name__ == "__main__":
    # 2. Initialize the Sweep
    
    sweep_id = wandb.sweep(sweep_config, project="pytorch-sqlite-sweeps")
    
    print(f"🚀 Starting sweep with ID: {sweep_id}")
    
    # 3. Run the Agent (Executes the training loop 5 times)
    #samples new HP from sweep_config for each run
    #calls wandb.init and injects sampled values into wandb.config
    wandb.agent(sweep_id, function=sweep_train, count=5)