import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertModel
# add the 'vision-transformer' directory to the Python path to allow importing ViT model
sys.path.append(os.path.abspath(".."))
from vision_transformer import VisionTransformer

class CLIPVisionEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # initialise vision transformer with placeholder values
        # these will be overwritten when i load pretrained weights later
        # since we are using the pretrained model, don't drop paths
        self.vit = VisionTransformer(
            classes = 10,
            drop_path_prob = 0.0
        )
        # bypass the classification head to allow projection to embedding dimension
        self.vit.CLS_head = nn.Identity()
        # projection of the 768-dimensional CLS token to the embedding dimension
        self.projection = nn.Linear(768, embed_dim)

    def forward(self, x):
        cls_token = self.vit(x)
        return self.projection(cls_token)


class CLIPTextEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.projection = nn.Linear(768, embed_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # extract the last hidden state of the CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_token)


class CLIP(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.vision_encoder = CLIPVisionEncoder(embed_dim)
        self.text_encoder = CLIPTextEncoder(embed_dim)
        # learnable logit scale parameter for contrastive loss
        # initialised to ln(1/0.07) as in the original CLIP paper
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
    
    def forward(self, images, input_ids, attention_mask):
        image_embeddings = self.vision_encoder(images)
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        # normalise embeddings to allow calculation of cosine similarity
        # here we use L2 norm
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        # initialise logit scale parameter to be positive by exponentiating
        # if negative, it would invert the similarities
        logit_scale = self.logit_scale.exp()
        # matrix multiplication of image embeds and transposed text embeds to get similarity
        # gives the most similar text for an image
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
        # since we use symmetric loss, we need transposed matrix to get the most similar image for a text
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text
