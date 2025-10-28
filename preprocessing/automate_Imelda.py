import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def automate_preprocessing(input_path, output_dir="../dataset_preprocessing"):
    df = pd.read_csv(input_path)
    print("Dataset berhasil dibaca!")
    
    print("\nCek data...")
    print("Missing values per kolom:\n", df.isnull().sum())
    print("Jumlah data duplikat:", df.duplicated().sum())
    df = df.drop_duplicates()
    
    X = df.drop("quality", axis=1)
    y = df["quality"]
    
    print("\nNormalisasi fitur...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    df_preprocessed = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "winequality_preprocessing.csv")
    df_preprocessed.to_csv(output_path, index=False)
    
    print(f"\n✅ Preprocessing otomatis selesai! File disimpan di {output_path}")
    return df_preprocessed


if __name__ == "__main__":
    automate_preprocessing("../dataset_raw/winequality.csv")
