import torch
from torchvision.transforms import v2

def train_epoch(model, dataloader, loss_criterion, optimiser, device, classes, with_cutmix=False):
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # initialise cutmix
    if with_cutmix:
        cutmix = v2.CutMix(num_classes=classes)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # apply CutMix augmentation before forward pass
        # this stitches patches of one image onto another and adjusts label percentages accordingly    
        if with_cutmix:
            inputs, labels = cutmix(inputs, labels)

        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        # since loss is averaged over batch, multiply by batch size to get total loss
        total_loss += loss.item() * inputs.size(0)

        if with_cutmix:
            # compute accuracy based on mixed labels (need to find argmax)
            preds = outputs.argmax(dim=1)
            label_max = labels.argmax(dim=1)
            correct_preds += (preds == label_max).sum().item()
        else:
            preds = outputs.argmax(dim=1)
            correct_preds += (preds == labels).sum().item()

        total_preds += labels.size(0)

    epoch_loss = total_loss / total_preds
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc

@torch.no_grad()
def eval_model(model, dataloader, loss_criterion, device):
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)

        preds = outputs.argmax(dim=1)
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = total_loss / total_preds
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc
    