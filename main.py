"""
define model pipeline function

"""

import wandb
import torch
import torch.nn as nn
import torch.optim as torch_optim
from torch.utils.data import DataLoader, random_split # <-- Added random_split
import os

from config import PARAMS
from src.utils import set_seed, get_device
from src.dataset import MNISTDatabaseDataset
from src.model import ConvNet
from src.trainer import train_model
from src.evaluator import test_model

os.environ["WANDB_HTTP_TIMEOUT"] = "300" 

def make(db_path, config, device):
    # 1. Load the Full Training Data (60,000 images)
    full_train_dataset = MNISTDatabaseDataset(db_path, split='train')

    # 2. Split it: 50k for Training, 10k for Validation
    # We use a fixed generator seed so the split is the same every time
    train_size = int(0.833 * len(full_train_dataset)) # approx 50,000
    val_size = len(full_train_dataset) - train_size   # approx 10,000

    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) 
    )

    # 3. Load the Test Data (10,000 images) - HELD OUT
    test_dataset = MNISTDatabaseDataset(db_path, split='test')

    # 4. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # 5. Model Setup
    model = ConvNet(config.kernels, config.classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch_optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, val_loader, test_loader, criterion, optimizer

def model_pipeline(hyperparameters=None):
    device = get_device()
    set_seed()

    # wandb.run is None if not doing a HP sweep
    if wandb.run is None:
        # not doing HP sweep, so use default hyperparameters
        run = wandb.init(project="pytorch-sqlite-ops", 
                         job_type = "training", 
                         config=hyperparameters,
                         name="training")
    
    config = run.config

    # This tells W&B: "This specific run depends on mnist-sqlite-data:latest"
    artifact = run.use_artifact(config.dataset_artifact, type="dataset")
    artifact_dir = artifact.download()

    # Resolve runtime path
    db_path = os.path.join(artifact_dir, "mnist.db")

    print("🔗 Run linked to dataset artifact")
    print(f"📦 Artifact: {config.dataset_artifact}")
    print(f"📁 DB path: {db_path}")

    # Get all 3 loaders
    model, train_loader, val_loader, test_loader, criterion, optimizer = make(db_path, config, device)
    
    # Pass train_loader AND val_loader to the trainer
    train_model(model, train_loader, val_loader, criterion, optimizer, config, device)
    
    # Evaluate on the unseen Test Set only at the very end
    test_model(model, test_loader, config, device, run)
    run.finish()
    return model

if __name__ == "__main__":
    try:
        model_pipeline(PARAMS)
    except Exception as e:
        print(f"An error occurred: {e}")