import os
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers import DistilBertTokenizer

class CLIPFlickr8k(Dataset):
    def __init__(self, datapath, annotations, transforms=None):
        self.datapath = datapath
        self.tokeniser = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        raw_annotations = defaultdict(list)
        with open(annotations, 'r') as f:
            for line in f:
                if '\t' in line:
                    img_id, caption = line.strip().split('\t', 1)
                    img_id = img_id.split('.jpg')[0] + '.jpg'
                    raw_annotations[img_id].append(caption)
                    
        self.annotations = {}
        self.image_ids = []
        
        for img_id, captions in raw_annotations.items():
            img_path = os.path.join(self.datapath, img_id)
            if os.path.exists(img_path):
                self.image_ids.append(img_id)
                self.annotations[img_id] = captions
                
        print(f"Dataloader initialised: Found {len(self.image_ids)} valid images.")

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = v2.Compose([
                v2.Resize((224, 224), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        img_id = self.image_ids[i]
        caption_set = self.annotations[img_id]
        
        img_path = os.path.join(self.datapath, img_id)
        image = Image.open(img_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)
        caption = random.choice(caption_set)
        tokenised_caption = self.tokeniser(caption, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
        # return tensors gives tensor with batch_dim 1, so we squeeze
        return {
            'image': image,
            'input_ids': tokenised_caption['input_ids'].squeeze(0),
            'attention_mask': tokenised_caption['attention_mask'].squeeze(0)
        }

