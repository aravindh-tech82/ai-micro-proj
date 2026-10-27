import os
import sys
import zipfile
import urllib.request
import pandas as pd
from tqdm import tqdm

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "raw"))

URLS = {
    "GTSRB-Training_fixed.zip": "https://huggingface.co/datasets/Zzyy2000/GTSRB/resolve/main/GTSRB-Training_fixed.zip",
    "GTSRB_Final_Test_Images.zip": "https://huggingface.co/datasets/Zzyy2000/GTSRB/resolve/main/GTSRB_Final_Test_Images.zip",
    "Test.csv": "https://huggingface.co/datasets/bazyl/GTSRB/resolve/main/Test.csv",
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, target_path):
    print(f"Downloading {os.path.basename(target_path)}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(target_path)) as t:
        urllib.request.urlretrieve(url, filename=target_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {os.path.basename(zip_path)} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # 1. Download training zip
    train_zip = os.path.join(RAW_DIR, "GTSRB-Training_fixed.zip")
    if not os.path.exists(train_zip):
        download_file(URLS["GTSRB-Training_fixed.zip"], train_zip)
    else:
        print("Training zip already downloaded.")

    # 2. Download test zip
    test_zip = os.path.join(RAW_DIR, "GTSRB_Final_Test_Images.zip")
    if not os.path.exists(test_zip):
        download_file(URLS["GTSRB_Final_Test_Images.zip"], test_zip)
    else:
        print("Test images zip already downloaded.")

    # 3. Extract zips into raw directory
    train_extract_check = os.path.join(RAW_DIR, "GTSRB", "Training")
    if not os.path.exists(train_extract_check):
        extract_zip(train_zip, RAW_DIR)
    else:
        print("Training images already extracted.")

    test_extract_check = os.path.join(RAW_DIR, "GTSRB", "Final_Test")
    if not os.path.exists(test_extract_check):
        extract_zip(test_zip, RAW_DIR)
    else:
        print("Test images already extracted.")

    # 4. Download and setup test annotations
    test_csv_path = os.path.join(RAW_DIR, "Test.csv")
    gt_csv_path = os.path.join(RAW_DIR, "GT-final_test.csv")
    
    if not os.path.exists(test_csv_path):
        download_file(URLS["Test.csv"], test_csv_path)

    # Also generate official GT-final_test.csv (semicolon delimited with PPM filenames)
    if not os.path.exists(gt_csv_path) and os.path.exists(test_csv_path):
        df_test = pd.read_csv(test_csv_path)
        # Standard columns: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
        df_gt = pd.DataFrame()
        df_gt["Filename"] = df_test["Path"].apply(lambda p: os.path.basename(p).replace(".png", ".ppm"))
        df_gt["Width"] = df_test["Width"]
        df_gt["Height"] = df_test["Height"]
        df_gt["Roi.X1"] = df_test["Roi.X1"]
        df_gt["Roi.Y1"] = df_test["Roi.Y1"]
        df_gt["Roi.X2"] = df_test["Roi.X2"]
        df_gt["Roi.Y2"] = df_test["Roi.Y2"]
        df_gt["ClassId"] = df_test["ClassId"]
        df_gt.to_csv(gt_csv_path, sep=";", index=False)
        print(f"Created official GT-final_test.csv with {len(df_gt)} rows at {gt_csv_path}")

    print("GTSRB raw dataset ready in:", RAW_DIR)

if __name__ == "__main__":
    main()
