"""Generate data (if needed), train the model, save artifacts.   Usage: python train.py"""
from nsre.config import DATASET_PATH, MODEL_PATH
from nsre.data import class_distribution, generate_dataset, save_dataset
from nsre.model import save_bundle, train_model

if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
    print(f"Dataset saved: {DATASET_PATH}  ({len(df)} rows)")
    print("Class distribution:\n", class_distribution(df))
    bundle = train_model(df)
    save_bundle(bundle)
    print(f"Model saved: {MODEL_PATH}  (train={bundle['train_size']}, test={bundle['test_size']})")
