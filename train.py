import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from dotenv import load_dotenv
from src.models import MedicalVLMExperiment
from src.dataset import CheXpertDataModule
from src.dataset_iu import IUChestDataModule, IUChestDataset

load_dotenv()

def main():
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    L.seed_everything(config['experiment']['seed'])
    
    # dm = CheXpertDataModule(config)
    dm = IUChestDataModule(config)
    model = MedicalVLMExperiment(config)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename=f"iu-llava-{config['experiment']['situation']}-{{epoch:02d}}",
        save_top_k=1,
        mode="min"
    )
    
    trainer = L.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback]
    )
    
    if config['experiment']['situation'] == "zero_shot":
        print("Zero Shot mode: Logic would go here (omitted for brevity)")
    else:
        trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()