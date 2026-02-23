import pandas as pd
import numpy as np
import os

def process_iu_csv(df, class_names):
    print(f"Mapping clinical tags for {len(df)} images...")
    
    # Expanded Mapping: 8 Target Classes + No Finding
    # We include common IU synonyms (like Infiltrate for Consolidation)
    iu_mapping = {
        'Atelectasis': ['Pulmonary Atelectasis'],
        'Cardiomegaly': ['Cardiomegaly'],
        'Consolidation': ['Consolidation', 'Infiltrate', 'Airspace Disease'],
        'Edema': ['Pulmonary Edema', 'Pulmonary Congestion'],
        'Pleural Effusion': ['Pleural Effusion', 'Costophrenic Angle'],
        'Pneumonia': ['Pneumonia'],
        'Granuloma': ['Calcified Granuloma', 'Granulomatous Disease'],
        'Emphysema': ['Emphysema', 'Pulmonary Emphysema'],
        'No Finding': ['normal']
    }

    # Initialize labels
    y = np.zeros((len(df), len(class_names)), dtype=int)
    
    for i, problem_str in enumerate(df['Problems']):
        if pd.isna(problem_str):
            continue
            
        found_problems = [p.strip() for p in str(problem_str).split(';')]
        
        for col_idx, cls_name in enumerate(class_names):
            target_terms = iu_mapping.get(cls_name, [])
            if any(term in found_problems for term in target_terms):
                y[i, col_idx] = 1
                
    label_cols = pd.DataFrame(y, columns=class_names)
    
    # Fix paths to point to data/iu-chest/images/
    image_paths = df['filename'].apply(lambda x: f"data/iu-chest/images/images_normalized/{x}").values
    processed_df = pd.concat([pd.Series(image_paths, name="image_path"), label_cols], axis=1)
    
    # Remove rows that are all zeros (images with no relevant findings)
    relevant_rows = (processed_df[class_names].sum(axis=1) > 0)
    final_df = processed_df[relevant_rows].copy()
    
    print(f"Kept {len(final_df)} relevant samples.")
    return final_df

def main():
    local_dir = "./data/iu-chest"
    proj_path = os.path.join(local_dir, "indiana_projections.csv")
    repo_path = os.path.join(local_dir, "indiana_reports.csv")
    
    if not os.path.exists(proj_path) or not os.path.exists(repo_path):
        print(f"Error: Required CSV files not found in {local_dir}")
        return

    # 1. Merge Projections and Reports
    proj_df = pd.read_csv(proj_path)
    repo_df = pd.read_csv(repo_path)
    df = pd.merge(proj_df, repo_df, on='uid')
    
    # 2. Filter for Frontal
    df = df[df['projection'] == 'Frontal'].copy()

    # 3. Define the 9 Columns (8 pathologies + 1 control)
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Pleural Effusion', 'Pneumonia', 'Granuloma', 'Emphysema', 'No Finding'
    ]

    # 4. Process
    full_dataset = process_iu_csv(df, class_names)

    # 5. Shuffle & Split (100 for Test)
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    test_df = full_dataset.head(100)
    train_df = full_dataset.tail(len(full_dataset) - 100)

    # 6. Save back to data/iu-chest
    train_out = os.path.join(local_dir, "train_split.csv")
    test_out = os.path.join(local_dir, "test_split.csv")
    
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"\nCSV Files saved in: {local_dir}")
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")
    print(f"Columns: {', '.join(class_names)}")

if __name__ == "__main__":
    main()