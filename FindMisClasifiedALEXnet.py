"""
find_misclassified.py

Scans your dataset, finds images AlexNet gets wrong, saves results to CSV.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import csv
import json
import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH = "imagenet_val_dataset"
OUTPUT_CSV   = "misclassified.csv"
# ─────────────────────────────────────────────────────────────────────────────

alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
alexnet.eval()
alex_preprocess = models.AlexNet_Weights.DEFAULT.transforms()

class_idx  = json.load(open("imagenet_class_index.json"))
idx2label  = np.array([class_idx[str(k)][1] for k in range(len(class_idx))])
true_class_nums = np.loadtxt("val.txt", dtype=int)

files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith(".JPEG")])

rows = []
for i, fname in enumerate(files):
    img        = Image.open(os.path.join(DATASET_PATH, fname)).convert("RGB")
    tensor     = alex_preprocess(img)
    true_label = int(true_class_nums[i])

    with torch.no_grad():
        out   = alexnet(tensor.unsqueeze(0))
        probs = F.softmax(out, dim=1)[0]
        pred  = out.argmax(dim=1).item()

    if pred != true_label:
        rows.append({
            "index":            i,
            "file":             fname,
            "true_class_idx":   true_label,
            "true_class_name":  idx2label[true_label],
            "pred_class_idx":   pred,
            "pred_class_name":  idx2label[pred],
            "true_confidence":  round(probs[true_label].item(), 4),
            "pred_confidence":  round(probs[pred].item(), 4),
        })
        print(f"  #{i:<4}  true: {idx2label[true_label]:<30}  got: {idx2label[pred]:<30}  conf: {probs[pred].item()*100:.1f}%")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone — {len(rows)} misclassified images saved to {OUTPUT_CSV}")