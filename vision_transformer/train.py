import os
import json
import argparse
import wandb

import time, csv, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models import VisionTransformer
from utils import setup_dataloaders, train_epoch, eval_model

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

def setup_DDP():
    init_process_group(backend="nccl")
    # retrieve id of the GPU to be assigned to current proc
    lrank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lrank)
    return lrank

def main():
    # parse json config to run specified experiment
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as cfg:
        config = json.load(cfg)

    set_seed(SEED)

    # for running on multiple GPUs
    local_rank = setup_DDP()
    global_rank = int(os.environ["RANK"])

    if global_rank == 0:
        wandb.init(project="vision-transformer", config=config, name=config.get("run", "vit-experiments"))

    imagedim = config.get("imagedim", 224)
    num_classes = config.get("num_classes", 10)
    epochs = config.get("epochs", 100)
    with_cutmix = config.get("with_cutmix", False)

    # setup dataloaders
    trainloader, testloader, trainsampler = setup_dataloaders(
        directory=config.get("directory", "../data"),
        imagedim=imagedim,
        batchsize=config.get("batchsize", 32),
        workers=8,
        seed=SEED,
        is_distributed=True
    )

    # initialise model
    model = VisionTransformer(
        classes=num_classes,
        drop_path_prob=config.get("drop_prob", 0.1),
        imagedim=imagedim
    )

    model.to(local_rank)

    loss_criterion = nn.CrossEntropyLoss()
    # AdamW as optimiser, with configurable weighted decay
    optimiser = optim.AdamW(
        model.parameters(),
        lr = config.get("lr", 0.001),
        weight_decay=config.get("weight_decay", 0.05)
    )

    warmup = config.get("warmup", 5)
    min_lr = config.get("min_lr", 0.000001)

    warmup_scheduler = LinearLR(optimiser, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    cosine_scheduler = CosineAnnealingLR(optimiser, T_max=(epochs-warmup), eta_min=min_lr)
    scheduler = SequentialLR(
        optimiser,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup]
    )

    # checkpoint loading
    start_epoch = 0
    run_name = config.get("run")
    checkpoint_path = config.get("checkpoint_path", f"checkpoints/{run_name}.pt")
    csv_path = config.get("csv_path", f"logs/{run_name}_metrics.csv")

    if global_rank == 0:
        # recursively create directories in path if they don't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(checkpoint_path):
        if global_rank == 0:
            # note that we wrap any logging in a global rank check to avoid duplicate logs
            print(f"Resuming training from last checkpoint: {checkpoint_path}...")
        # load to specific GPU
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{local_rank}")
        # load state dicts saved from last checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # load saved RNG states for deterministic execution
        torch.set_rng_state(checkpoint["rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        # resume training from last epoch
        start_epoch = checkpoint["epoch"] + 1
    else:
        if global_rank == 0:
            print("Starting training from scratch...")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "lr", "trainloss", "trainacc", "testloss", "testacc"])

    # putting this in a list is just a quirk of legacy syntax
    model = DDP(model, device_ids=[local_rank])

    start_time = time.time()
    # training
    for epoch in range(epochs):
        # to ensure data is shuffled differently during each epoch
        trainsampler.set_epoch(epoch)

        trainloss, trainacc = train_epoch(
            model=model,
            dataloader=trainloader,
            loss_criterion=loss_criterion,
            optimiser=optimiser,
            device=local_rank,
            classes=num_classes,
            with_cutmix=with_cutmix
        )

        testloss, testacc = eval_model(
            model=model,
            dataloader=testloader,
            loss_criterion=loss_criterion,
            device=local_rank
        )

        scheduler.step()

        # logging metrics to wandb
        if global_rank == 0:
            current_lr = optimiser.param_groups[0]["lr"]
            print(f"epoch: {epoch+1}/{epochs}\tcurrent LR: {current_lr:.6f}\ttrainloss: {trainloss:.4f}\ttrainacc: {trainacc*100:.2f}%\ttestacc: {testacc*100:.2f}%")

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{current_lr:.6f}", f"{trainloss:.4f}", f"{trainacc*100:.2f}", f"{testloss:.4f}", f"{testacc*100:.2f}"])

            wandb.log({
                "epoch": epoch+1,
                "lr": current_lr,
                "trainloss": trainloss,
                "trainacc": trainacc,
                "testloss": testloss,
                "testacc": testacc
            })

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            }
            torch.save(checkpoint, checkpoint_path)

    # training done
    if global_rank == 0:
        wandb.finish()

    destroy_process_group()

if __name__ == "__main__":
    main()
