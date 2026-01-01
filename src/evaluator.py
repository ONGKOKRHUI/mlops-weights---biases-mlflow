import os
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

def test_model(model, test_loader, config, device, run): # <--- Add 'run' argument
    print("🧪 Running final evaluation...")

    # ---- Evaluate ----
    metrics = evaluate(model, test_loader, device, split="test")
    
    # Use the passed 'run' object for logging
    run.log(metrics) 

    # Promote metrics to summary
    for k, v in metrics.items():
        run.summary[k] = v

    print(f"Final Test Metrics: {metrics}")

    # -------------------------
    # Export ONNX
    # -------------------------
    model.eval()
    os.makedirs("model", exist_ok=True)

    dummy_input = next(iter(test_loader))[0].to(device)
    
    # Fix: Use run.id from the passed object
    onnx_filename = f"mnist_{run.id}.onnx"
    onnx_path = os.path.join("model", onnx_filename)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
    )
    
    assert os.path.exists(onnx_path), f"File not found: {onnx_path}"
    
    # -------------------------
    # Candidate model artifact
    # -------------------------
    candidate_artifact = wandb.Artifact(
        name="mnist-cnn-candidate",
        type="model",
        metadata={
            "run_id": run.id,
            "test_accuracy": metrics["test_accuracy"],
            "architecture": config.architecture,
        },
    )

    # Fix: Use abspath to ensure Windows doesn't get confused
    candidate_artifact.add_file(os.path.abspath(onnx_path))

    print("📦 Uploading candidate model artifact...")
    
    # CRITICAL FIX: Use run.log_artifact instead of wandb.log_artifact
    run.log_artifact(candidate_artifact)
    print("✅ Candidate model logged")

    # -------------------------
    # Best model promotion
    # -------------------------
    current_val = run.summary.get("val_accuracy")
    if current_val is not None:
        best_val = run.summary.get("best_val_accuracy", 0.0)

        if current_val > best_val:
            run.summary["best_val_accuracy"] = current_val

            best_artifact = wandb.Artifact(
                name="mnist-cnn-best",
                type="model",
                description="Best model within this run",
                metadata={
                    "val_accuracy": current_val,
                    "test_accuracy": metrics["test_accuracy"],
                },
            )

            # Link the SAME file path
            best_artifact.add_file(os.path.abspath(onnx_path))

            print("🏆 Logging new best model (run-level)...")
            run.log_artifact(best_artifact)
            print("✅ New best model logged")

        else:
            print("ℹ️ Model not better than current run best")
    