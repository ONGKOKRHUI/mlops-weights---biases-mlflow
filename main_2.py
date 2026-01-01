"""
main.py
"""

import wandb
import torch
import onnx
import torch.nn as nn
import torch.optim as torch_optim
import torch.onnx  # Added for ONNX export
from torch.utils.data import DataLoader, random_split
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score # Added for metrics
import traceback

from config import PARAMS
from src.utils import set_seed, get_device
from src.dataset import MNISTDatabaseDataset
from src.model import ConvNet
from src.trainer import train_model
# from src.evaluator import test_model <-- Removed

os.environ["WANDB_HTTP_TIMEOUT"] = "300" 

def evaluate(model, loader, device, split="val"):
    """
    Calculates comprehensive metrics. 
    Moved here from evaluator.py to support demodularization.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Move to CPU for Scikit-Learn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    metrics = {
        f"{split}_accuracy": accuracy,
        f"{split}_precision": precision,
        f"{split}_recall": recall,
        f"{split}_f1": f1
    }
    
    model.train()
    return metrics

def make(db_path, config, device):
    # 1. Load the Full Training Data (60,000 images)
    full_train_dataset = MNISTDatabaseDataset(db_path, split='train')

    # 2. Split it: 50k for Training, 10k for Validation
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

    run = wandb.init(project="pytorch-sqlite-ops", 
                        job_type = "training", 
                        config=hyperparameters,
                        name="training")
    
    config = run.config

    # Download Dataset Artifact
    artifact = run.use_artifact(config.dataset_artifact, type="dataset")
    artifact_dir = artifact.download()
    db_path = os.path.join(artifact_dir, "mnist.db")

    print("🔗 Run linked to dataset artifact")
    print(f"📦 Artifact: {config.dataset_artifact}")
    print(f"📁 DB path: {db_path}")

    # Get all 3 loaders
    model, train_loader, val_loader, test_loader, criterion, optimizer = make(db_path, config, device)
    
    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, config, device)
    
    # ==========================================
    # FINAL EVALUATION & ARTIFACT LOGGING
    # (Formerly test_model)
    # ==========================================
    print("🧪 Running final evaluation...")

    # 1. Evaluate
    metrics = evaluate(model, test_loader, device, split="test")
    
    # Log metrics to W&B
    run.log(metrics) 
    for k, v in metrics.items():
        run.summary[k] = v

    print(f"Final Test Metrics: {metrics}")

    # 2. Export ONNX
    # model.eval()
    # os.makedirs("model", exist_ok=True)

    # dummy_input = next(iter(test_loader))[0].to(device)
    # model_filename = f"mnist_{run.id}.onnx"
    # model_path = os.path.join("model", model_filename)

    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     model_path,
    #     input_names=["input"],
    #     output_names=["output"],
    # )
    
    # # checking of onnx file
    # # 1. Check file exists
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"ONNX file was not created: {model_path}")

    # # 2. Try loading and checking the ONNX model
    # try:
    #     onnx_model = onnx.load(model_path)          # Load the ONNX model
    #     onnx.checker.check_model(onnx_model)       # Validate the model
    # except onnx.onnx_cpp2py_export.checker.ValidationError as e:
    #     raise RuntimeError(f"ONNX model is invalid: {e}")
    # print(f"✅ ONNX model exported and verified: {model_path}")
    
    #pytorch save try
    os.makedirs("model", exist_ok=True)
    model_path = "model/mock_mnist_model.pt"
    # model_path = os.path.abspath(model_path) # Removed to match upload_model.py behavior
    torch.save(model.state_dict(), model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not created: {model_path}")
    print(f"✅ Mock model saved at {model_path}")
    artifact_name = f"mnist-cnn-candidate-{run.id}"

    # 3. Log Candidate Model Artifact
    candidate_artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description="Candidate model within this run",
        metadata={
            "run_id": run.id,
            "test_accuracy": metrics["test_accuracy"],
            "architecture": config.architecture,
        },
    )

    candidate_artifact.add_file(model_path)

    print("📦 Uploading candidate model artifact...")
    run.log_artifact(candidate_artifact)
    print("✅ Candidate model logged")

    # # 4. Check for Best Model Promotion
    # current_val = run.summary.get("val_accuracy")
    # if current_val is not None:
    #     best_val = run.summary.get("best_val_accuracy", 0.0)

    #     if current_val > best_val:
    #         run.summary["best_val_accuracy"] = current_val

    #         best_artifact = wandb.Artifact(
    #             name="mnist-cnn-best",
    #             type="model",
    #             description="Best model within this run",
    #             metadata={
    #                 "val_accuracy": current_val,
    #                 "test_accuracy": metrics["test_accuracy"],
    #             },
    #         )

    #         best_artifact.add_file(model_path)

    #         print("🏆 Logging new best model (run-level)...")
    #         run.log_artifact(best_artifact)
    #         print("✅ New best model logged")
    #     else:
    #         print("ℹ️ Model not better than current run best")

    run.finish()
    return model

if __name__ == "__main__":
    try:
        model_pipeline(PARAMS)
    except Exception:
        traceback.print_exc()