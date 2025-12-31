import wandb
from tqdm.auto import tqdm
from src.evaluator import evaluate

def train_batch(images, labels, model, optimizer, criterion, device):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def train_model(model, train_loader, val_loader, criterion, optimizer, config, device):
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    example_ct = 0
    batch_ct = 0
    
    for epoch in tqdm(range(config.epochs)):
        model.train()
        for _, (images, labels) in enumerate(train_loader):
            loss = train_batch(images, labels, model, optimizer, criterion, device)
            example_ct += len(images)
            batch_ct += 1
            
            # Log raw training loss frequently (for debugging)
            if ((batch_ct + 1) % 25) == 0:
                wandb.log({"epoch": epoch, "batch_loss": float(loss)}, step=example_ct)
        
        # --- End of Epoch: Compare Train vs Validation ---
        
        # 1. Validation Metrics
        val_metrics = evaluate(model, val_loader, device, split="val")
        
        # 2. Training Metrics (Calculate on a subset or full train set to compare apples-to-apples)
        # For speed, we usually trust the batch_loss, but for a clean graph, let's eval on train set
        # (Optional: to save time, you can skip this or use a subset)
        train_metrics = evaluate(model, train_loader, device, split="train")
        
        # Merge metrics
        logs = {**train_metrics, **val_metrics, "epoch": epoch}
        
        # Log to W&B
        wandb.log(logs, step=example_ct)
        
        print(f"Epoch {epoch} | Train Acc: {train_metrics['train_accuracy']:.2%} | Val Acc: {val_metrics['val_accuracy']:.2%}")