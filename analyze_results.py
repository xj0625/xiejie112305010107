import pandas as pd

for train_dir in ['train', 'train-2', 'train-3']:
    try:
        df = pd.read_csv(f'runs/detect/{train_dir}/results.csv')
        print(f'\n=== {train_dir} ===')
        print(f'Total epochs: {len(df)}')
        best_idx = df['metrics/mAP50(B)'].idxmax()
        print(f'Best mAP50: {df["metrics/mAP50(B)"].iloc[best_idx]:.4f} at epoch {best_idx+1}')
        print(f'Final mAP50: {df["metrics/mAP50(B)"].iloc[-1]:.4f}')
        print(f'Final Precision: {df["metrics/precision(B)"].iloc[-1]:.4f}, Recall: {df["metrics/recall(B)"].iloc[-1]:.4f}')
    except Exception as e:
        print(f'\n=== {train_dir} ===')
        print(f'Error: {e}')