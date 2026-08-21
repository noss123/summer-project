import os
import json
import argparse
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from models import VisionTransformer, StochasticDepth
from utils import setup_dataloaders, train_epoch, eval_model

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
        batchsize=config.get("batchsize", 128),
        workers=8,
        is_distributed=True
    )

    # initialise model
    model = VisionTransformer(
        classes=num_classes,
        drop_path_prob=config.get("drop_prob", 0.1),
        imagedim=imagedim
    )

    model.to(local_rank)
    # putting this in a list is just a quirk of legacy syntax
    model.DDP(model, device_ids=[local_rank])

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

    # training
    for epoch in range(epochs):
        # to ensure data is shuffled differently during each epoch
        trainsampler.set_epoch(epoch)

        trainloss, trainacc = train_epoch(
            model=model,
            dataloader=trainloader,
            criterion=loss_criterion,
            optimiser=optimiser,
            device=local_rank,
            classes=num_classes,
            with_cutmix=with_cutmix
        )

        testloss, testacc = eval_model(
            model=model,
            dataloader=testloader,
            criterion=loss_criterion,
            device=local_rank
        )

        scheduler.step()

        # logging metrics to wandb
        if global_rank == 0:
            current_lr = optimiser.param_groups[0]["lr"]
            print(f"epoch: {epoch+1}/{epochs}\tcurrent LR: {current_lr:.6f}\ttrainloss: {trainloss:.4f}\ttrainacc: {trainacc*100:.2f}%\ttestacc: {testacc*100:.2f}%")

            wandb.log({
                "epoch": epoch+1,
                "trainloss": trainloss,
                "trainacc": trainacc,
                "testloss": testloss,
                "testacc": testacc,
                "lr": current_lr
            })

    # training done
    if global_rank == 0:
        wandb.finish()

    destroy_process_group()

if __name__ == "__main__":
    main()