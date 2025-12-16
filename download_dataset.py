#To download the SONICs (25K AI song dataset) & the GTZAN (1K real song dataset), run the following commands:
# 1) pip install -r requirements.txt
# 2) python download_dataset.py

import kagglehub

def main() -> None:
    print("Downloading datasets from Kaggle...")
    print("-" * 50)

    # Download SONICS Dataset (AI-generated vs human-generated music)
    print("\n1. Downloading SONICS dataset...")
    sonics_path = kagglehub.dataset_download("awsaf49/sonics-dataset")
    print(f"   ✓ SONICS dataset path: {sonics_path}")

    # Download GTZAN Dataset (music genre classification)
    print("\n2. Downloading GTZAN dataset...")
    gtzan_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    print(f"   ✓ GTZAN dataset path: {gtzan_path}")

    print("\n" + "-" * 50)
    print("All datasets downloaded successfully!")
    print("\nDataset locations:")
    print(f"  - SONICS: {sonics_path}")
    print(f"  - GTZAN:  {gtzan_path}")

if __name__ == "__main__":
    main()


