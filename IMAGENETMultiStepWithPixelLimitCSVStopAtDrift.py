import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ── change these ──────────────────────────
alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
alexnet.eval()
alex_preprocess = models.AlexNet_Weights.DEFAULT.transforms()

IMAGE_INDEX          = 18482
TARGET_CLASS         = 547      # ImageNet class index (e.g. 65 = sea_snake)
MAX_PIXELS_SIMUL     = 50
RERANK_EVERY         = 1
MAX_STEPS            = 5000
CONFIDENCE_THRESHOLD = 0.80
STEP_SIZE            = 0.005
EPSILON              = 4.0
# ─────────────────────────────────────────

class_idx  = json.load(open("imagenet_class_index.json"))
idx2label  = np.array([class_idx[str(k)][1] for k in range(len(class_idx))])
true_class_nums = np.loadtxt("val.txt", dtype=int)

torch.manual_seed(42)


def get_pixel_gradients(image, model, target_class):
    model.eval()
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)
    img  = image.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(img), torch.tensor([target_class]))
    model.zero_grad()
    loss.backward()
    return img.grad.squeeze().detach()

def apply_importance_filter(important_pixel_indexes, grads):
    flat_grads = grads.reshape(-1)
    ans = torch.zeros_like(flat_grads)
    ans[important_pixel_indexes] = flat_grads[important_pixel_indexes]
    return ans.reshape_as(grads)

def get_N_highest_importance(grads, max_simultaneous_pixels):
    top_N_indexes = np.argpartition(-grads.reshape(-1), -max_simultaneous_pixels)[-max_simultaneous_pixels:]
    ans = apply_importance_filter(top_N_indexes, grads)
    return ans, top_N_indexes

def pixel_attack(image, model, true_label, target_class, initial_pred, max_pixels=500, max_steps=500, step_size=0.003, epsilon=0.03, rerank_every=10, confidence_threshold=0.80):
    model.eval()

    original  = image.clone()
    perturbed = image.clone().unsqueeze(0)

    true_confidences   = []
    target_confidences = []
    wrong_confidences  = []
    pred_history       = []
    importance_history = []
    flip_step          = None
    cur_targets        = []
    cur_target_grads   = []

    for step in range(max_steps):
        perturbed = perturbed.detach().requires_grad_(True)

        loss = F.cross_entropy(model(perturbed), torch.tensor([target_class]))
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grads = perturbed.grad

            if step % rerank_every == 0:
                cur_target_grads, cur_targets = get_N_highest_importance(grads.detach().clone(), MAX_PIXELS_SIMUL)
            else:
                cur_target_grads = apply_importance_filter(cur_targets, grads.detach().clone())

            target_importance = cur_target_grads.abs().squeeze().detach().clone()
            importance_history.append(target_importance)

            perturbed = perturbed - step_size * cur_target_grads.sign()
            perturbed = torch.max(perturbed, original.unsqueeze(0) - epsilon)
            perturbed = torch.min(perturbed, original.unsqueeze(0) + epsilon)

            out         = model(perturbed)
            probs       = F.softmax(out, dim=1)
            true_conf   = probs[0, true_label].item()
            target_conf = probs[0, target_class].item()
            wrong_conf  = probs[0, initial_pred].item()
            pred        = out.argmax(dim=1).item()

        true_confidences.append(true_conf)
        target_confidences.append(target_conf)
        wrong_confidences.append(wrong_conf)
        pred_history.append(pred)

        if flip_step is None and pred == target_class:
            flip_step = step + 1
            print(f"  Flipped to {idx2label[target_class]} at step {flip_step} — continuing...")

        if target_conf >= confidence_threshold:
            print(f"  Reached {confidence_threshold*100:.0f}% confidence at step {step+1} — stopping.")
            break

    noise_map  = (perturbed.squeeze() - original.squeeze()).detach().numpy()
    final_img  = perturbed.squeeze().detach()

    early_importance = importance_history[0]
    mid_importance   = importance_history[len(importance_history) // 2]
    final_importance = importance_history[-1]

    return (final_img, true_confidences, target_confidences, wrong_confidences, pred_history,
            flip_step, noise_map, early_importance, mid_importance, final_importance)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def show_rgb(ax, tensor):
    img_np = tensor.numpy().transpose(1, 2, 0)
    img_np = img_np * _IMAGENET_STD + _IMAGENET_MEAN
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(img_np)

def show_map(ax, arr):
    if arr.ndim == 3:
        arr = arr.mean(0) if isinstance(arr, np.ndarray) else arr.numpy().mean(0)
    ax.imshow(arr, cmap='hot')

def find_example_image(class_index, true_class_nums, files, dataset_path):
    for i, label in enumerate(true_class_nums):
        if label == class_index and i < len(files):
            img = Image.open(os.path.join(dataset_path, files[i])).convert("RGB")
            return alex_preprocess(img)
    return None


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), "imagenet_val_dataset")
    files = sorted([f for f in os.listdir(file_path) if f.endswith(".JPEG")])
    img_file = files[IMAGE_INDEX]

    img   = Image.open(os.path.join(file_path, img_file)).convert("RGB")
    image = alex_preprocess(img)

    true_label = true_class_nums[IMAGE_INDEX]

    with torch.no_grad():
        initial_pred = alexnet(image.unsqueeze(0)).argmax(dim=1).item()

    print(f"Image #{IMAGE_INDEX}  ({img_file})")
    print(f"  true: {idx2label[true_label]}  |  target: {idx2label[TARGET_CLASS]}  |  initial pred: {idx2label[initial_pred]}")
    print(f"Running for up to {MAX_STEPS} steps...\n")

    result = pixel_attack(
        image, alexnet, true_label, TARGET_CLASS,
        initial_pred,
        max_pixels=MAX_PIXELS_SIMUL, max_steps=MAX_STEPS,
        step_size=STEP_SIZE, epsilon=EPSILON,
        rerank_every=RERANK_EVERY, confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    (perturbed_img, true_confs, target_confs, wrong_confs, pred_history,
     flip_step, noise_map, early_imp, mid_imp, final_imp) = result

    actual_steps = len(true_confs)
    final_pred   = pred_history[-1]

    with torch.no_grad():
        out   = alexnet(perturbed_img.unsqueeze(0))
        probs = F.softmax(out, dim=1)[0]
        top5  = probs.topk(5)
    print(f"\n── Verification (fresh forward pass on perturbed image) ──")
    for conf, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        marker = " ◄ TARGET" if idx == TARGET_CLASS else (" ◄ TRUE" if idx == true_label else "")
        print(f"  {idx2label[idx]:<30} {conf*100:5.1f}%{marker}")
    print()

    print(f"Final prediction after {actual_steps} steps: {idx2label[final_pred]}")
    if flip_step:
        print(f"First flipped to {idx2label[TARGET_CLASS]} at step {flip_step}")
    else:
        print(f"Never reached target class {idx2label[TARGET_CLASS]}")

    initial_pred_img = find_example_image(initial_pred, true_class_nums, files, file_path)
    target_img       = find_example_image(TARGET_CLASS, true_class_nums, files, file_path)

    fig = plt.figure(figsize=(16, 13))

    ax0 = fig.add_subplot(4, 4, 1)
    show_rgb(ax0, image)
    ax0.set_title(f"Original  (true: {idx2label[true_label]})")
    ax0.axis('off')

    ax1 = fig.add_subplot(4, 4, 2)
    show_rgb(ax1, perturbed_img)
    ax1.set_title(f"After {actual_steps} steps  →  {idx2label[final_pred]}")
    ax1.axis('off')

    ax2 = fig.add_subplot(4, 4, 3)
    ax2.imshow(noise_map.mean(0), cmap='bwr', vmin=-EPSILON, vmax=EPSILON)
    ax2.set_title("Cumulative pixel drift\n(red=brighter, blue=darker)")
    ax2.axis('off')

    ax3 = fig.add_subplot(4, 4, 4)
    show_rgb(ax3, image)
    ax3.imshow(abs(noise_map.mean(0)) > 0.01, cmap='Reds', alpha=0.5)
    ax3.set_title("Pixels that moved")
    ax3.axis('off')

    ax4 = fig.add_subplot(4, 4, 5)
    show_map(ax4, early_imp)
    ax4.set_title("Importance — step 1")
    ax4.axis('off')

    ax5 = fig.add_subplot(4, 4, 6)
    show_map(ax5, mid_imp)
    ax5.set_title(f"Importance — step {actual_steps // 2}")
    ax5.axis('off')

    ax6 = fig.add_subplot(4, 4, 7)
    show_map(ax6, final_imp)
    ax6.set_title(f"Importance — step {actual_steps}")
    ax6.axis('off')

    ax7 = fig.add_subplot(4, 4, 8)
    all_importance = torch.stack([early_imp, mid_imp, final_imp]).mean(dim=0)
    show_map(ax7, all_importance)
    ax7.set_title("Average importance")
    ax7.axis('off')

    ax_init = fig.add_subplot(4, 4, 9)
    if initial_pred_img is not None:
        show_rgb(ax_init, initial_pred_img)
    ax_init.set_title(f"Example: {idx2label[initial_pred]}\n(initial prediction)")
    ax_init.axis('off')

    ax_tgt = fig.add_subplot(4, 4, 10)
    if target_img is not None:
        show_rgb(ax_tgt, target_img)
    ax_tgt.set_title(f"Example: {idx2label[TARGET_CLASS]}\n(attack target)")
    ax_tgt.axis('off')

    ax8 = fig.add_subplot(4, 1, 4)
    xs = range(1, actual_steps + 1)
    ax8.plot(xs, true_confs,   color='steelblue', linewidth=2, label=f'Confidence in {idx2label[true_label]} (true)')
    ax8.plot(xs, target_confs, color='tomato',    linewidth=2, label=f'Confidence in {idx2label[TARGET_CLASS]} (target)')
    ax8.plot(xs, wrong_confs,  color='orange',    linewidth=2, linestyle='--', label=f'Confidence in {idx2label[initial_pred]} (initial wrong pred)')

    prev = pred_history[0]
    seg_start = 0
    colors = {true_label: 'lightblue', TARGET_CLASS: 'lightsalmon'}
    for i, p in enumerate(pred_history):
        if p != prev or i == actual_steps - 1:
            color = colors.get(prev, 'lightyellow')
            ax8.axvspan(seg_start, i, alpha=0.2, color=color)
            seg_start = i
            prev = p

    if flip_step:
        ax8.axvline(x=flip_step, color='black', linestyle='--', label=f'First flip at step {flip_step}')

    ax8.legend(fontsize=9)
    ax8.set_xlabel("Step")
    ax8.set_ylabel("Model confidence")
    ax8.set_title("Full run — background shading shows what model predicts at each step")
    ax8.set_ylim(0, 1)

    plt.suptitle(
        f"Run stopped at step {actual_steps}  —  {idx2label[true_label]} → target {idx2label[TARGET_CLASS]}  "
        f"(step_size={STEP_SIZE}, epsilon={EPSILON})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()