import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
# Change 1: Use Llava-specific class
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

def run_zero_shot(config_path):
    load_dotenv()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Change 2: Ensure you point to the LLaVA-Med ID in your YAML
    model_name = config['model'].get('id', 'microsoft/llava-med-v1.5-mistral-7b')
    print(f"Loading LLaVA-Med: {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    device = model.device

    csv_path = config['data']['csv_path']
    test_df = pd.read_csv(csv_path)

    # Using your IU Unique classes or CheXpert classes
    # class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Pleural Effusion', 'Pneumonia', 'Granuloma', 'Emphysema', 'No Finding'
    ]
    results = []
    all_preds = {cls: [] for cls in class_names}
    all_trues = {cls: [] for cls in class_names}

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            continue

        result_row = {'image_path': image_path}

        for cls in class_names:
            true_label = int(row.get(cls, 0))
            result_row[f"{cls}_True"] = true_label
            all_trues[cls].append(true_label)

            # Change 3: LLaVA-Med v1.5 specific prompt format
            # LLaVA requires the <image> token explicitly in the string
            prompt = f"USER: <image>\nIs there {cls} in this X-ray? Answer 1 or 0. ASSISTANT:"
            
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                # generate only a few tokens for 1/0
                generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

            # Change 4: Slicing the output to get only the new tokens
            input_token_len = inputs.input_ids.shape[1]
            output_text = processor.batch_decode(generated_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

            # Simple parsing for LLaVA's direct response
            pred = 1 if "1" in output_text else 0

            result_row[f"{cls}_Pred"] = pred
            all_preds[cls].append(pred)

        results.append(result_row)

    # --- 4. SAVE & REPORT ---
    output_csv = config['experiment']['output_csv']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)

    print("\n" + "=" * 40)
    print(f"AVERAGE ACCURACY: {np.mean([accuracy_score(all_trues[c], all_preds[c]) for c in class_names]):.4f}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/zero_shot_llava.yaml")
    args = parser.parse_args()
    run_zero_shot(args.config)