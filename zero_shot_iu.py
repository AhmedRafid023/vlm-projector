import argparse
import yaml
import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

def run_zero_shot(config_path):
    # --- 0. LOAD ENV ---
    load_dotenv()

    # --- 1. SETUP ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config['model'].get('id', 'google/medgemma-4b-it')
    print(f"Loading Model: {model_name}...")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # low_cpu_mem_usage=True is critical for L4 (24GB) stability
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()

    device = next(model.parameters()).device

    # --- 2. LOAD DATA ---
    csv_path = config['data']['csv_path']
    print(f"Reading Test CSV: {csv_path}")
    test_df = pd.read_csv(csv_path)

    # Updated to the 9 classes defined in your IU-Chest script
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Pleural Effusion', 'Pneumonia', 'Granuloma', 'Emphysema', 'No Finding'
    ]
    
    results = []
    all_preds = {cls: [] for cls in class_names}
    all_trues = {cls: [] for cls in class_names}

    print(f"Starting Zero-Shot Evaluation on {len(test_df)} images...")

    # --- 3. INFERENCE LOOP ---
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = row['image_path']

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print(f"Skipping missing image: {image_path}")
            continue

        result_row = {'image_path': image_path}

        for cls in class_names:
            true_label = int(row.get(cls, 0))
            result_row[f"{cls}_True"] = true_label
            all_trues[cls].append(true_label)

            # --- PROMPT CONSTRUCTION ---
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Is there {cls} in this X-ray? Answer 1 or 0."}
                    ]
                }
            ]

            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # --- PRECISION HANDLING ---
            # Casting pixel_values to bfloat16 to match model weights
            inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            with torch.no_grad():
                # max_new_tokens=10 allows for short conversational answers
                generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)

            # --- ROBUST PARSING ---
            # input_len ensures we don't accidentally "see" the 1/0 in the prompt text
            input_len = inputs["input_ids"].shape[1]
            output_text = processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]

            ans = output_text.strip().lower()
            # Catching "1", "yes", or affirmative responses
            pred = 1 if ("1" in ans or "yes" in ans) else 0

            result_row[f"{cls}_Pred"] = pred
            all_preds[cls].append(pred)

        results.append(result_row)
        
        # Periodic memory cleanup for long inference runs
        if idx % 20 == 0:
            torch.cuda.empty_cache()

    # --- 4. SAVE & REPORT ---
    output_csv = config['experiment']['output_csv']
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)

    print("\n" + "=" * 40)
    print(f"RESULTS SAVED: {output_csv}")
    print("=" * 40)

    print(f"{'Class':<20} | {'Accuracy':<10}")
    print("-" * 33)
    avg_acc = 0
    for cls in class_names:
        acc = accuracy_score(all_trues[cls], all_preds[cls])
        print(f"{cls:<20} | {acc:.4f}")
        avg_acc += acc

    print("-" * 33)
    print(f"{'AVERAGE':<20} | {avg_acc/len(class_names):.4f}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/zero_shot.yaml")
    args = parser.parse_args()

    run_zero_shot(args.config)