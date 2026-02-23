import argparse
import yaml
import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv  # Added this to read .env


def run_zero_shot(config_path):
    # --- 0. LOAD ENV TOKEN ---
    load_dotenv()

    # --- 1. SETUP ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config['model'].get('id', 'google/medgemma-4b-it')
    print(f"Loading Model: {model_name}...")

    # AutoProcessor and Model will now automatically find 'HF_TOKEN' from environment
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    device = model.device

    # --- 2. LOAD DATA ---
    csv_path = config['data']['csv_path']
    print(f"Reading Test CSV: {csv_path}")
    test_df = pd.read_csv(csv_path)

    print(f"Starting Zero-Shot Evaluation on {len(test_df)} images...")

    # class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    class_names = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
            'Pleural Effusion', 'Pneumonia', 'Granuloma', 'Emphysema', 'No Finding'
        ]
    results = []

    all_preds = {cls: [] for cls in class_names}
    all_trues = {cls: [] for cls in class_names}

    # --- 3. INFERENCE LOOP ---
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = row['image_path']

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping missing image: {image_path}")
            continue

        result_row = {'image_path': image_path}

        for cls in class_names:
            true_label = int(row.get(cls, 0))
            result_row[f"{cls}_True"] = true_label
            all_trues[cls].append(true_label)


            # --- PREDICTION ---
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

            inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)

            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parsing logic for 1/0
            if "Answer 1 or 0." in output_text:
                ans = output_text.split("Answer 1 or 0.")[-1].strip()
            else:
                ans = output_text[-5:].strip()

            pred = 1 if "1" in ans else 0

            # # --- PREDICTION ---
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "image"},
            #             {"type": "text", "text": f"Analyze this chest X-ray. Search for signs of {cls}. "
            #                                     "First, describe what you see, then conclude with "
            #                                     "'Final Answer: 1' if present or 'Final Answer: 0' if not."}
            #         ]
            #     }
            # ]

            # # Increase max_new_tokens to allow for the reasoning
            # generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            # output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # # Update parsing to look for the final answer
            # if "Final Answer:" in output_text:
            #     ans_part = output_text.split("Final Answer:")[-1].strip()
            #     pred = 1 if "1" in ans_part else 0
            # else:
            #     # Fallback to last character logic
            #     pred = 1 if "1" in output_text[-5:] else 0

            result_row[f"{cls}_Pred"] = pred
            all_preds[cls].append(pred)

        results.append(result_row)

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
