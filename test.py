# import torch
# import yaml
# import pandas as pd
# import argparse
# import os
# import numpy as np
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from transformers import AutoProcessor
# from src.models import MedicalVLMExperiment
# from src.dataset import CheXpertDataset
# from src.dataset_iu import IUChestDataModule, IUChestDataset



# def run_test(config_path):
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)

#     # -------------------------
#     # Load Model
#     # -------------------------
#     ckpt_path = config['test']['checkpoint_path']
#     if not os.path.exists(ckpt_path):
#         print(f"Checkpoint not found: {ckpt_path}")
#         return

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = MedicalVLMExperiment.load_from_checkpoint(
#         ckpt_path,
#         config=config,
#         map_location=device
#     )
#     model.to(device).eval().freeze()

#     # -------------------------
#     # Processor
#     # -------------------------
#     processor = AutoProcessor.from_pretrained(
#         config['model']['id'],
#         trust_remote_code=True
#     )

#     # -------------------------
#     # Dataset
#     # -------------------------
#     # test_ds = CheXpertDataset(config['data']['csv_path'], processor)
#     test_ds = IUChestDataset(config['data']['csv_path'], processor)
#     loader = DataLoader(
#         test_ds,
#         batch_size=config['data']['batch_size']
#     )

#     # -------------------------
#     # results
#     # -------------------------
#     results = []
#     # class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
#     class_names = [
#         'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
#         'Pleural Effusion', 'Pneumonia', 'Granuloma', 'Emphysema', 'No Finding'
#     ]

#     all_preds = []
#     all_true = []

#     with torch.no_grad():
#         for batch in tqdm(loader):
#             px = batch['pixel_values'].to(device)
#             true = batch['label'].cpu().numpy()
#             paths = batch['image_path']

#             logits = model(px)
#             probs = torch.sigmoid(logits)
#             preds = (probs > config['test']['threshold']).int().cpu().numpy()

#             all_preds.append(preds)
#             all_true.append(true)

#             for i in range(len(true)):
#                 row = {"image_path": paths[i]}
#                 for idx, name in enumerate(class_names):
#                     row[f"{name}_True"] = int(true[i][idx])
#                     row[f"{name}_Pred"] = int(preds[i][idx])
#                 results.append(row)

#     # -------------------------
#     # Save CSV
#     # -------------------------
#     os.makedirs(os.path.dirname(config['test']['output_csv']), exist_ok=True)

#     df = pd.DataFrame(results)
#     df.to_csv(config['test']['output_csv'], index=False)

#     print(f"Results saved to {config['test']['output_csv']}")

#     # =====================================================
#     # ✅ ACCURACY COMPUTATION
#     # =====================================================
#     all_preds = np.vstack(all_preds)
#     all_true = np.vstack(all_true)

#     correct = (all_preds == all_true)

#     overall_acc = correct.mean()
#     print(f"\nOverall Accuracy: {overall_acc:.4f}")

#     print("\nPer-class Accuracy:")
#     for i, name in enumerate(class_names):
#         acc = correct[:, i].mean()
#         print(f"{name}: {acc:.4f}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="configs/test.yaml")
#     args = parser.parse_args()

#     run_test(args.config)








# LLava-med Test

import torch
import yaml
import pandas as pd
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from src.models import MedicalVLMExperiment
from src.dataset import CheXpertDataset
from src.dataset_iu import IUChestDataModule, IUChestDataset


def run_test(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. PRE-LOAD CLEANUP
    # Ensuring the L4 VRAM is as fresh as possible
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # -------------------------
    # 2. LOAD MODEL (Meta-Tensor & OOM Fix)
    # -------------------------
    ckpt_path = config['test']['checkpoint_path']
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    
    # CRITICAL: map_location="cpu" prevents the "doubling" of VRAM.
    # The weights load into RAM first, then Lightning populates the GPU-resident model.
    model = MedicalVLMExperiment.load_from_checkpoint(
        ckpt_path,
        config=config,
        map_location="cpu" 
    )
    
    # CRITICAL: Do NOT call model.to(device). 
    # self.full_model in models.py already uses device_map="auto".
    model.eval().freeze()
    
    # Dynamically find the device of the first parameter to send inputs there.
    main_device = next(model.parameters()).device
    print(f"Model successfully sharded. Primary device: {main_device}")

    # -------------------------
    # 3. PROCESSOR & DATASET
    # -------------------------
    processor = AutoProcessor.from_pretrained(
        config['model']['id'],
        trust_remote_code=True
    )

    # test_ds = CheXpertDataset(config['data']['csv_path'], processor)
    test_ds = IUChestDataset(config['data']['csv_path'], processor)
    loader = DataLoader(
        test_ds,
        batch_size=config['data']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        shuffle=False
    )

    # -------------------------
    # 4. INFERENCE LOOP
    # -------------------------
    results = []
    # class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Pleural Effusion', 'Pneumonia', 'Granuloma', 'Emphysema', 'No Finding'
    ]

    all_preds = []
    all_true = []

    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # Convert inputs to bfloat16 to match model weights and save memory
            px = batch['pixel_values'].to(main_device, dtype=torch.bfloat16)
            true = batch['label'].cpu().numpy()
            paths = batch['image_path']

            logits = model(px)
            probs = torch.sigmoid(logits)
            
            threshold = config['test'].get('threshold', 0.5)
            preds = (probs > threshold).int().cpu().numpy()

            all_preds.append(preds)
            all_true.append(true)

            for i in range(len(true)):
                row = {"image_path": paths[i]}
                for idx, name in enumerate(class_names):
                    row[f"{name}_True"] = int(true[i][idx])
                    row[f"{name}_Pred"] = int(preds[i][idx])
                    # Storing probability is useful for future AUROC calculation
                    row[f"{name}_Prob"] = float(probs[i][idx])
                results.append(row)

    # -------------------------
    # 5. SAVE & REPORT
    # -------------------------
    output_csv = config['test']['output_csv']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Detailed results saved to {output_csv}")

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)

    correct = (all_preds == all_true)
    overall_acc = correct.mean()
    
    print("\n" + "="*30)
    print(f"OVERALL ACCURACY: {overall_acc:.4f}")
    print("-" * 30)
    for i, name in enumerate(class_names):
        acc = correct[:, i].mean()
        print(f"{name:<20}: {acc:.4f}")
    print("="*30 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test.yaml")
    args = parser.parse_args()

    run_test(args.config)