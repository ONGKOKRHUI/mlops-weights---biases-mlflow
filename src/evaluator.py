import torch
import torch.onnx
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate(model, loader, device, split="val"):
    """Calculates comprehensive metrics."""
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

def test_model(model, test_loader, config, device):
    """Final test: logs metrics and registers model artifact."""
    print("🧪 Running final evaluation...")
    metrics = evaluate(model, test_loader, device, split="test")
    
    # Log all metrics at once
    wandb.log(metrics)
    
    print(f"Final Test Metrics: {metrics}")

    # Export to ONNX
    dummy_input = next(iter(test_loader))[0].to(device)
    onnx_path = config.model_path
    torch.onnx.export(model, dummy_input, onnx_path)
    
    # --- MODEL REGISTRY ---
    # Create an Artifact instead of just saving a file
    model_artifact = wandb.Artifact(
        name="mnist-cnn-model", 
        type="model",
        description="Trained MNIST CNN model"
    )
    model_artifact.add_file(onnx_path)
    wandb.log_artifact(model_artifact)
    print("✅ Model registered to W&B Artifacts")