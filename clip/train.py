import os
import json, argparse
import random
import numpy as np
import wandb

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from transformers import get_cosine_schedule_with_warmup

from models import CLIP
from utils import CLIPFlickr8k, train_step, evaluate_loss, evaluate_CLIP, qualitative_predictions

SEED = 26

def set_seed(seed):
    # enforce deterministic execution for reproducibility
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # forces cuDNN to use deterministic algorithms only
    torch.backends.cudnn.deterministic = True
    # disables cuDNN autotuner that selects optimal algorithm based on hardware
    torch.backends.cudnn.benchmark = False

def build_param_groups(model, config):
    return [
        {"params": model.vision_encoder.vit.parameters(), "lr": config.get("vit_lr", 5e-6)},
        {"params": model.text_encoder.bert.parameters(), "lr": config.get("bert_lr", 2e-6)},
        {"params": list(model.vision_encoder.projection.parameters()) + list(model.text_encoder.projection.parameters()) + [model.logit_scale],
         "lr": config.get("head_lr", 1e-4)},
    ]

# helper function to freeze backbones of vision/text encoders
def set_backbone_trainable(model, is_trainable):
    for p in model.vision_encoder.vit.parameters():
        p.requires_grad = is_trainable
    for p in model.text_encoder.bert.parameters():
        p.requires_grad = is_trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading config from: {args.config}", flush=True)
    with open(args.config, 'r') as cfg:
        config = json.load(cfg)

    print("Setting seed...", flush=True)
    set_seed(SEED)

    print("Initialising wandb...", flush=True)
    wandb.init(project='clip', name=config['run'], config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    print("Setting up image transforms...", flush=True)
    transforms = v2.Compose([
        v2.Resize((224, 224), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Initialising Flickr8k dataset...", flush=True)
    dataset = CLIPFlickr8k(
        datapath=config['datapath'],
        annotations=config['annotations'],
        transforms=transforms
    )

    print("Setting up train/test splits and dataloaders...", flush=True)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        drop_last=False
    )

    print("Initialising CLIP model...", flush=True)
    model = CLIP(embed_dim=config.get('embed_dim', 256)).to(device)
    print("CLIP model initialised successfully.", flush=True)

    vit_pretrained_path = config.get('vit_pretrained_path', None)
    if vit_pretrained_path and os.path.exists(vit_pretrained_path):
        print(f"Loading ViT pretrained weights from {vit_pretrained_path}...", flush=True)
        checkpoint = torch.load(vit_pretrained_path, map_location=device)
        model.vision_encoder.vit.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Successfully loaded pretrained ViT backbone from {vit_pretrained_path}")
    else:
        print(f"No ViT checkpoint found at {vit_pretrained_path}. Vision encoder initialised randomly.")
    
    optimiser = optim.AdamW(
        build_param_groups(model, config), 
        weight_decay=config.get("weight_decay", 0.01)
    )

    epochs = config.get('epochs', 10)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(config.get('warmup_ratio', 0.05) * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # to store best model based on validation loss
    checkpoint_path = config.get('checkpoint_path', 'checkpoints/clip.pt')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_loss = float('inf')

    freeze_epochs = config.get('freeze_epochs', 2)
    for epoch in range(epochs):
        set_backbone_trainable(model, is_trainable=(epoch >= freeze_epochs))

        train_loss = train_step(model, train_dataloader, optimiser, device, scheduler)
        val_loss = evaluate_loss(model, val_dataloader, device)

        recall_metrics, _ = evaluate_CLIP(model, val_dataloader, device)
        pred_table = qualitative_predictions(model, val_dataloader, dataset.tokeniser, device, epoch)
        
        current_lr = optimiser.param_groups[0]["lr"]
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            **recall_metrics,
            "logit_scale": model.logit_scale.exp().item(),
            "qualitative_predictions": pred_table
        })

        print(f"epoch: {epoch+1} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | lr: {current_lr:.6f} | i2t_recall@1: {recall_metrics['i2t_recall@1']:.2f} | t2i_recall@1: {recall_metrics['t2i_recall@1']:.2f} | scale: {model.logit_scale.exp().item():.4f}", flush=True)

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved new best model with validation loss {best_val_loss:.4f}")

    print("Training complete. Running final evaluation on the held-out test set...", flush=True)
    best_checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_recall_metrics, _ = evaluate_CLIP(model, test_dataloader, device)
    final_test_logs = {f"test_{k}": v for k, v in test_recall_metrics.items()}
    wandb.log(final_test_logs)

    print(f"final i2t_recall@1 [test]: {test_recall_metrics['i2t_recall@1']:.2f} | final t2i_recall@1 [test]: {test_recall_metrics['t2i_recall@1']:.2f}", flush=True)

    wandb.finish()

if __name__ == "__main__":
    main()
            
