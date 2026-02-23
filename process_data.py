import pandas as pd
import numpy as np
import argparse
import os
import shutil
import kagglehub

def process_csv(df, policy, class_names, split_name):
    print(f"Processing {split_name} split...")
    
    # 1. Filter Frontal (if column exists)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal'].copy()
    
    df.fillna(0, inplace=True)
    
    # 2. Fix Paths
    clean_paths = []
    raw_paths = df["Path"].to_numpy()
    
    for p in raw_paths:
        if "train/" in p:
            suffix = p.split("train/", 1)[1]
            clean_paths.append(f"data/dataset/train/{suffix}")
        elif "valid/" in p:
            suffix = p.split("valid/", 1)[1]
            clean_paths.append(f"data/dataset/valid/{suffix}")
        else:
            clean_paths.append(p)

    # 3. Process Labels
    y_df = df[class_names]
    class_ones = ['Atelectasis', 'Cardiomegaly']
    
    # Initialize the full array to store all labels
    y = np.zeros((len(y_df), len(class_names)), dtype=int)
    y_values = y_df.to_numpy()
    
    for i in range(len(y_values)):
        row_vals = y_values[i]
        row_labels = [] 
        for col_idx, cls_name in enumerate(class_names):
            curr_val = row_vals[col_idx]
            feat_val = 0
            if curr_val:
                curr_val = float(curr_val)
                if curr_val == 1: feat_val = 1
                elif curr_val == -1: # Uncertain policy
                    if policy == "ones": feat_val = 1
                    elif policy == "zeroes": feat_val = 0
                    elif policy == "mixed": feat_val = 1 if cls_name in class_ones else 0
            row_labels.append(feat_val)
        
        # Store the row in the main array
        y[i] = row_labels
        
    # 4. Construct Result DataFrame
    clean_df = pd.DataFrame(y, columns=class_names)
    clean_df.insert(0, "image_path", clean_paths)
    
    return clean_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="mixed")
    args = parser.parse_args()

    # --- 1. DATA DOWNLOAD ---
    local_dataset_dir = "./data/dataset"
    # Check for valid.csv as a proxy for "dataset exists"
    if not os.path.exists(os.path.join(local_dataset_dir, "valid.csv")):
        print("Downloading CheXpert via KaggleHub...")
        try:
            cache_path = kagglehub.dataset_download("ashery/chexpert")
            if os.path.exists(local_dataset_dir): shutil.rmtree(local_dataset_dir)
            shutil.copytree(cache_path, local_dataset_dir)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading: {e}")
            return

    # --- 2. LOCATE FILES ---
    train_src = os.path.join(local_dataset_dir, "train.csv")
    valid_src = os.path.join(local_dataset_dir, "valid.csv")
    
    # Fallback search
    if not os.path.exists(train_src):
        for root, _, files in os.walk(local_dataset_dir):
            if "train.csv" in files: train_src = os.path.join(root, "train.csv")
            if "valid.csv" in files: valid_src = os.path.join(root, "valid.csv")

    if not os.path.exists(train_src) or not os.path.exists(valid_src):
        raise FileNotFoundError(f"Could not find train.csv or valid.csv in {local_dataset_dir}")

    # --- 3. PROCESS ---
    class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    # A. Process Training Data
    df_train = pd.read_csv(train_src)
    clean_train = process_csv(df_train, args.policy, class_names, "TRAIN")
    
    # --- LIMIT TO 5000 SAMPLES ---
    if len(clean_train) > 5000:
        print(f"Limiting training set from {len(clean_train)} to 5000 random samples...")
        clean_train = clean_train.sample(n=5000, random_state=42).reset_index(drop=True)
    
    # B. Process Validation Data (To be used as TEST)
    # We generally keep the Full Validation set for accurate metrics
    df_valid = pd.read_csv(valid_src)
    clean_test = process_csv(df_valid, args.policy, class_names, "VALIDATION (Used as TEST)")

    # --- 4. SAVE ---
    os.makedirs("./data", exist_ok=True)
    
    train_out = "./data/train_split.csv"
    clean_train.to_csv(train_out, index=False)
    
    test_out = "./data/test_split.csv"
    clean_test.to_csv(test_out, index=False)

    print(f"------------------------------------------------")
    print(f"Success!")
    print(f"Training Data: {train_out} ({len(clean_train)} images)")
    print(f"Test Data:     {test_out} ({len(clean_test)} images)")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    main()