import pandas as pd

df = pd.read_csv("misclassified.csv")

interesting = df[
    (df["pred_confidence"] > 0.80) &
    (df["pred_confidence"] < 0.95) &
    (df["true_confidence"] < 0.10)
].sample(10, random_state=42)

print(interesting[["index", "true_class_idx", "pred_confidence"]].to_string(index=False))