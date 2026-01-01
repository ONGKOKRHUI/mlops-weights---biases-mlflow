import wandb
import os

# 1. Config matching your project
# (Increase timeout to fix the "graphql request timed out" error you saw)
os.environ["WANDB_HTTP_TIMEOUT"] = "300" 

def upload():
    # Initialize a run JUST for data upload
    run = wandb.init(
        project="pytorch-sqlite-sweeps",     #project="pytorch-sqlite-ops"
        job_type="upload_dataset",
        name="upload_mnist_data"
    )
    
    # Path to your file
    db_path = "data/mnist.db" # <--- Verify this path matches your local folder

    if os.path.exists(db_path):
        print(f"✅ Found file at {db_path}")
        
        # Create the artifact
        artifact = wandb.Artifact(
            name="mnist-sqlite-data", 
            type="dataset",
            description="MNIST database file for training"
        )
        
        # Add file
        artifact.add_file(db_path)
        
        # Log and explicit WAIT
        print("⏳ Uploading to W&B... this may take a minute...")
        run.log_artifact(artifact)
        artifact.wait() # <--- Crucial: forces script to pause until upload is 100% done
        
        print("🎉 Upload Success! Check the 'dataset' tab in W&B now.")
    else:
        print(f"❌ Error: Could not find file at {db_path}")

    # Explicitly finish the run
    run.finish()

if __name__ == "__main__":
    upload()


#################################
"""
When to use References
wandb.Artifact.add_reference("link to s3 bucket")
According to the documentation, this is best used when your files are stored in external object storage like an Amazon S3 bucket. This is useful when:

Data is huge: You have a 1TB dataset that is too large to upload to W&B.

Data is restricted: The data cannot leave your corporate cloud bucket for security reasons.

Duplicate Avoidance: You don't want to create copies of data that is already safely stored in the cloud.
"""
##############################################