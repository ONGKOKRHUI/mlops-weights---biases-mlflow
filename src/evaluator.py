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
    print("🧪 Running final evaluation...")

    # ---- Evaluate ----
    metrics = evaluate(model, test_loader, device, split="test")
    wandb.log(metrics)

    # Move final metrics to summary
    for k, v in metrics.items():
        wandb.run.summary[k] = v

    print(f"Final Test Metrics: {metrics}")

    # ---- Export ONNX (run-unique path) ----
    dummy_input = next(iter(test_loader))[0].to(device)
    onnx_path = f"model/mnist_{wandb.run.id}.onnx"
    torch.onnx.export(model, dummy_input, onnx_path)

    # ---- Candidate model artifact ----
    candidate_artifact = wandb.Artifact(
        name="mnist-cnn-candidate",
        type="model",
        metadata={
            "val_accuracy": wandb.run.summary.get("val_accuracy"),
            "test_accuracy": metrics["test_accuracy"]
        }
    )
    candidate_artifact.add_file(onnx_path)
    wandb.log_artifact(candidate_artifact)

    # ---- Best-model promotion ----
    current_val = wandb.run.summary.get("val_accuracy", 0)
    best_so_far = wandb.run.summary.get("best_val_accuracy", 0)

    if current_val > best_so_far:
        wandb.run.summary["best_val_accuracy"] = current_val

        best_artifact = wandb.Artifact(
            name="mnist-cnn-best",
            type="model",
            description="Best model from sweep"
        )
        best_artifact.add_file(onnx_path)
        wandb.log_artifact(best_artifact)

        print("🏆 New best model logged!")

    else:
        print("ℹ️ Model not better than current best")
