import torch
import torch.nn.functional as F
import wandb

def CLIP_loss(logits_per_image, logits_per_text):
    batchsize = logits_per_image.shape[0]
    labels = torch.arange(batchsize, device=logits_per_image.device)
    # compute cross entropy loss for image-to-text
    loss_i = F.cross_entropy(logits_per_image, labels)
    # compute cross entropy loss for text-to-image
    loss_t = F.cross_entropy(logits_per_text, labels)
    # average of these gives InfoNCE loss for CLIP
    return (loss_i + loss_t) / 2.0


def train_step(model, dataloader, optimiser, device, scheduler=None, max_norm=1.0):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimiser.zero_grad()
        logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
        loss = CLIP_loss(logits_per_image, logits_per_text)
        loss.backward()
        
        # to guard against exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimiser.step()

        if scheduler is not None:
            scheduler.step()
        
        # to prevent the logit scale from growing too large, we clamp it to a max value of ln(100)
        with torch.no_grad():
            model.logit_scale.clamp_(0, torch.log(torch.tensor(100.0)))

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
        
        loss = CLIP_loss(logits_per_image, logits_per_text)
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_CLIP(model, dataloader, device):
    model.eval()
    all_image_features = []
    all_text_features = []

    for batch in dataloader:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        image_features = model.vision_encoder(image)
        text_features = model.text_encoder(input_ids, attention_mask)

        # normalise using L2 norm, since we skipped the CLIP forward pass
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        all_image_features.append(image_features)
        all_text_features.append(text_features)
    
    # concatenate all features into a single tensor
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)

    similarity_matrix = all_image_features @ all_text_features.T

    # calculate recall@k for k=1,5,10
    num_samples = similarity_matrix.shape[0]
    targets = torch.arange(num_samples, device=device)
    recall_at_k = {}

    for k in [1, 5, 10]:
        # image-to-text recall (i2t)
        _, topk_text_indices = similarity_matrix.topk(k, dim=1)
        i2t_correct = (topk_text_indices == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        recall_at_k[f"i2t_recall@{k}"] = i2t_correct * 100.0
        # text-to-image recall (t2i)
        _, topk_image_indices = similarity_matrix.T.topk(k, dim=1)
        t2i_correct = (topk_image_indices == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        recall_at_k[f"t2i_recall@{k}"] = t2i_correct * 100.0

    return recall_at_k, similarity_matrix


@torch.no_grad()
def qualitative_predictions(model, dataloader, tokeniser, device, epoch, num_samples=5):
    model.eval()
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    image_features = model.vision_encoder(images)
    text_features = model.text_encoder(input_ids, attention_mask)

    # normalise using L2 norm
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    similarity_matrix = image_features @ text_features.T

    columns = ['image', 'ground_truth', 'top_1_prediction', 'top_2_prediction', 'top_3_prediction']
    table = wandb.Table(columns=columns)

    decoded_captions = tokeniser.batch_decode(input_ids, skip_special_tokens=True)

    for i in range(min(num_samples, images.size(0))):
        image = images[i].cpu()
        # undo the normalisation for displaying the image
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = torch.clamp(image, 0, 1)
        wandb_image = wandb.Image(image)

        ground_truth = decoded_captions[i]
        _, top3_indices = similarity_matrix[i].topk(3)
        top1_pred = decoded_captions[top3_indices[0]]
        top2_pred = decoded_captions[top3_indices[1]]
        top3_pred = decoded_captions[top3_indices[2]]

        table.add_data(wandb_image, ground_truth, top1_pred, top2_pred, top3_pred)

    return table
    