import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import lightning as L
from transformers import AutoProcessor
import os

class CheXpertDataset(Dataset):
    def __init__(self, data_source, processor):
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source
            
        self.processor = processor
        self.label_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        
        self.project_root = os.getcwd()
        self.local_data_root = os.path.join(self.project_root, "data", "dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        
        # Fallback Logic
        if not os.path.exists(image_path):
             if "train/" in image_path:
                suffix = image_path.split("train/", 1)[1]
                image_path = os.path.join(self.local_data_root, "train", suffix)
             elif "valid/" in image_path:
                suffix = image_path.split("valid/", 1)[1]
                image_path = os.path.join(self.local_data_root, "valid", suffix)

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        labels = row[self.label_cols].values.astype(float)
        
        # --- FIX: Use chat template to generate the correct <image> token string ---
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        
        # This generates the exact special token string the model expects (e.g. <image> or <loc_image> etc)
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        
        # Now pass that validated string
        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(labels, dtype=torch.float32),
            "image_path": image_path 
        }

class CheXpertDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config['model']['id'], 
            trust_remote_code=True
        )

    def setup(self, stage=None):
        full_df = pd.read_csv(self.config['data']['train_csv'])
        
        # Internal Split
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(full_df) * 0.9)
        
        train_df = full_df.iloc[:split_idx]
        val_df = full_df.iloc[split_idx:]
        
        # (Print statements removed as requested)

        self.train_ds = CheXpertDataset(train_df, self.processor)
        self.val_ds = CheXpertDataset(val_df, self.processor)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config['data']['batch_size'], num_workers=4)