import os
import torch
import torch.nn as nn
import wandb

# Optional: increase timeout (safe to keep)
os.environ["WANDB_HTTP_TIMEOUT"] = "300"


# -------------------------
# Mock model definition
# -------------------------
class MockMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def upload():
    # -------------------------
    # Init a run JUST for upload
    # -------------------------
    run = wandb.init(
        project="pytorch-sqlite-ops",
        job_type="upload_model",
        name="upload_mock_model"
    )

    # -------------------------
    # Create a mock model file
    # -------------------------
    os.makedirs("model", exist_ok=True)
    model_path = "model/mock_mnist_model.pt"

    model = MockMNISTModel()
    torch.save(model.state_dict(), model_path)

    print(f"✅ Mock model saved at {model_path}")

    # -------------------------
    # Create model artifact
    # -------------------------
    artifact = wandb.Artifact(
        name="mnist-cnn-mock",
        type="model",
        description="Mock MNIST CNN model for testing artifact workflows",
        metadata={
            "framework": "pytorch",
            "input_shape": [1, 28, 28],
            "num_classes": 10,
            "mock": True
        }
    )

    artifact.add_file(model_path)

    # -------------------------
    # Log artifact
    # -------------------------
    print("⏳ Uploading model artifact to W&B...")
    run.log_artifact(artifact)

    print("🎉 Upload complete! Check the Artifacts tab.")

    # -------------------------
    # Finish run (this finalizes upload)
    # -------------------------
    run.finish()


if __name__ == "__main__":
    upload()
