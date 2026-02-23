import pandas as pd
import numpy as np
import os
import shutil
import kagglehub

def process_iu_csv(df, class_names):
    print("Mapping IU 'Problems' to binary labels...")
    
    # Mapping for target classes
    iu_mapping = {
        'Atelectasis': 'Pulmonary Atelectasis',
        'Cardiomegaly': 'Cardiomegaly',
        'Consolidation': 'Consolidation',
        'Edema': 'Pulmonary Edema',
        'Pleural Effusion': 'Pleural Effusion',
        'No Finding': 'normal'
    }

    y = np.zeros((len(df), len(class_names)), dtype=int)
    
    for i, problem_str in enumerate(df['Problems']):
        if pd.isna(problem_str):
            continue
            
        found_problems = [p.strip() for p in str(problem_str).split(';')]
        
        for col_idx, cls_name in enumerate(class_names):
            target_term = iu_mapping.get(cls_name)
            if target_term in found_problems:
                y[i, col_idx] = 1
                
    clean_df = pd.DataFrame(y, columns=class_names)
    
    # --- FIXED IMAGE PATHS ---
    # Resulting path: data/iu-chest/images/XXXX_frontal.png
    clean_df.insert(0, "image_path", df['filename'].apply(lambda x: f"data/iu-chest/images/{x}"))
    
    return clean_df

def main():
    # 1. SET DIRECTORY
    local_dir = "./data/iu-chest"
    os.makedirs(local_dir, exist_ok=True)

    # 2. DOWNLOAD DATA
    # This downloads the raddar version which contains projections and reports
    if not os.path.exists(os.path.join(local_dir, "indiana_reports.csv")):
        print("Downloading IU Chest X-Ray via KaggleHub...")
        cache_path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
        
        # Move files to local_dir
        for item in os.listdir(cache_path):
            s = os.path.join(cache_path, item)
            d = os.path.join(local_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("Download and Move complete.")

    # 3. LOAD & MERGE
    proj_df = pd.read_csv(os.path.join(local_dir, "indiana_projections.csv"))
    repo_df = pd.read_csv(os.path.join(local_dir, "indiana_reports.csv"))

    # Link reports to filenames via 'uid'
    df = pd.merge(proj_df, repo_df, on='uid')
    
    # Keep only Frontal views for consistency with CheXpert
    df = df[df['projection'] == 'Frontal'].copy()

    # 4. PROCESS LABELS
    class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'No Finding']
    full_dataset = process_iu_csv(df, class_names)

    # 5. SHUFFLE & SPLIT
    # 200 for Test, the rest for Train
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    test_df = full_dataset.head(200)
    train_df = full_dataset.tail(len(full_dataset) - 200)

    # 6. SAVE IN DATA/IU-CHEST
    train_out = os.path.join(local_dir, "train_split.csv")
    test_out = os.path.join(local_dir, "test_split.csv")
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"\nCSV Files saved in: {local_dir}")
    print(f"Sample Path in CSV: {test_df['image_path'].iloc[0]}")
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

if __name__ == "__main__":
    main()