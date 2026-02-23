import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import lightning as L
from transformers import AutoProcessor
import os

class IUChestDataset(Dataset):
    def __init__(self, data_source, processor):
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source
            
        self.processor = processor
        # Updated to the 9 columns we defined for IU-Chest
        self.label_cols = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
            'Pleural Effusion', 'Pneumonia', 'Granuloma', 'Emphysema', 'No Finding'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        
        # IU paths are already fixed in the CSV, but we use a basic check
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Fallback for missing images
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        labels = row[self.label_cols].values.astype(float)
        
        # Generate model-specific special tokens (e.g., <image>)
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        
        inputs = self.processor(text=prompt_text, images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(labels, dtype=torch.float32),
            "image_path": image_path 
        }

class IUChestDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config['model']['id'], 
            trust_remote_code=True
        )

    def setup(self, stage=None):
        # Pointing to the IU-specific CSVs
        if stage == 'fit' or stage is None:
            full_df = pd.read_csv(self.config['data']['train_csv'])
            # We skip the internal split if you want to use the full train_split.csv
            self.train_ds = IUChestDataset(full_df, self.processor)
            
            # If you have a separate validation set, load it here, 
            # otherwise we sample from train
            val_size = int(len(full_df) * 0.1)
            self.val_ds = IUChestDataset(full_df.iloc[:val_size], self.processor)
            
        if stage == 'test':
            test_df = pd.read_csv(self.config['data']['test_csv'])
            self.test_ds = IUChestDataset(test_df, self.processor)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['data']['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config['data']['batch_size'], num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config['data']['batch_size'], num_workers=4)