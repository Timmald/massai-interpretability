import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ── change these ──────────────────────────
pretrained_model = "data/lenet_mnist_model.pth"
IMAGE_INDEX      = 0
TARGET_CLASS     = 3
MAX_PIXELS_SIMUL = 50
RERANK_EVERY    = 1
MAX_STEPS        = 5000
STEP_SIZE        = 0.0005
EPSILON          = 1
# ─────────────────────────────────────────


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1    = nn.Conv2d(1, 32, 3, 1)
        self.conv2    = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1      = nn.Linear(9216, 128)
        self.fc2      = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


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
    flat_grads = grads.view(-1)
    ans = torch.zeros_like(flat_grads)
    ans[important_pixel_indexes]=flat_grads[important_pixel_indexes]
    return ans.reshape_as(grads)

def get_N_highest_importance(grads, max_simultaneous_pixels):
    # might need to swap the sign here, depen
    top_N_indexes = np.argpartition(-grads.view(-1), -max_simultaneous_pixels)[-max_simultaneous_pixels:]
    ans = apply_importance_filter(top_N_indexes, grads)
    return ans, top_N_indexes

def pixel_attack(image, model, true_label, target_class, max_pixels=500, max_steps=500, step_size = 0.003, epsilon=0.03, rerank_every=10):
    """
    Targeted greedy pixel attack with periodic re-ranking.
    Every `rerank_every` steps, recompute gradients from the current
    perturbed image so the attack adapts as the image changes.
    Stops when model predicts target_class.
    """
    model.eval()

    original  = image.clone()
    perturbed = image.clone().unsqueeze(0)
    
    flat_perturbed   = perturbed.view(-1)
    flat_noise       = torch.zeros(784)
    already_flipped  = set()

    true_confidences   = []
    target_confidences = []
    flip_k             = None
    flip_pred          = None
    flat_grads         = None
    cur_targets        = []
    cur_target_grads   = []

    true_confidences   = []
    target_confidences = []
    pred_history       = []   # track what the model says each step
    importance_history = []
    flip_step          = None

    for step in range(max_steps):
        perturbed = perturbed.detach().requires_grad_(True)
        perturbed_shape = perturbed.shape
        #grads      = get_pixel_gradients(perturbed, model, target_class)
        
        loss = F.cross_entropy(model(perturbed), torch.tensor([target_class]))
        model.zero_grad()
        loss.backward()
        
        # Recompute most important pixels from current perturbed state every rerank_every steps
        
        with torch.no_grad():
            grads = perturbed.grad
            #print(perturbed.grad)
            # Recompute most important pixels from current perturbed state every rerank_every steps
            if step % rerank_every == 0:
                cur_target_grads, cur_targets = get_N_highest_importance(grads.detach().clone(), MAX_PIXELS_SIMUL)
            else:
                cur_target_grads = apply_importance_filter(cur_targets, grads.detach().clone())
            
            target_importance = cur_target_grads.abs().squeeze().detach().clone()
            importance_history.append(target_importance)
            
            perturbed = perturbed - step_size * cur_target_grads.sign()
            perturbed = torch.max(perturbed, original.unsqueeze(0) - epsilon)
            perturbed = torch.min(perturbed, original.unsqueeze(0) + epsilon)
            perturbed = torch.clamp(perturbed, 0, 1)

            out         = model(perturbed)
            probs       = F.softmax(out, dim=1)
            true_conf   = probs[0, true_label].item()
            target_conf = probs[0, target_class].item()
            pred        = out.argmax(dim=1).item()

        true_confidences.append(true_conf)
        target_confidences.append(target_conf)
        pred_history.append(pred)
        # Note flip point but keep going
        if flip_step is None and pred == target_class:
            flip_step = step + 1
            print(f"  Flipped to {target_class} at step {flip_step} — continuing...")
    
    noise_map = (perturbed.squeeze() - original.squeeze()).detach().numpy()
    final_img = perturbed.squeeze().detach()

    early_importance = importance_history[0]
    mid_importance   = importance_history[len(importance_history) // 2]
    final_importance = importance_history[-1]

    return (final_img, true_confidences, target_confidences, pred_history,
            flip_step, noise_map, early_importance, mid_importance, final_importance)

    """for k in range(max_pixels):

        # Recompute gradients from current perturbed state every N steps
        if k % rerank_every == 0:
            grads      = get_pixel_gradients(flat_perturbed.view(1, 28, 28), model, target_class)
            flat_grads = grads.view(-1)
            # Zero out already-attacked pixels so we don't re-flip them
            for i in already_flipped:
                flat_grads[i] = 0.0
            ranked     = torch.argsort(flat_grads.abs(), descending=True)
            rank_queue = [i.item() for i in ranked if i.item() not in already_flipped]

        if not rank_queue:
            break

        idx  = rank_queue.pop(0)
        sign = flat_grads[idx].sign().item()

        # Push pixel to its extreme in the direction that helps target_class
        flat_perturbed[idx] = 1.0 if sign > 0 else 0.0
        flat_noise[idx]     = sign
        already_flipped.add(idx)

        with torch.no_grad():
            out         = model(flat_perturbed.view(1, 1, 28, 28))
            probs       = F.softmax(out, dim=1)
            true_conf   = probs[0, true_label].item()
            target_conf = probs[0, target_class].item()
            pred        = out.argmax(dim=1).item()

        true_confidences.append(true_conf)
        target_confidences.append(target_conf)

        if pred == target_class:
            flip_k    = k + 1
            flip_pred = pred
            break"""
    

    noise_map = flat_noise.view(28, 28).numpy()
    final_img = flat_perturbed.view(1, 28, 28).detach()
    return final_img, true_confidences, target_confidences, flip_k, flip_pred, noise_map


if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
    model.eval()

    transform    = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

    image, label = test_dataset[IMAGE_INDEX]

    with torch.no_grad():
        initial_pred = model(image.unsqueeze(0)).argmax(dim=1).item()

    print(f"Image #{IMAGE_INDEX}  |  true: {label}  |  target: {TARGET_CLASS}  |  initial pred: {initial_pred}")
    print(f"Running for all {MAX_STEPS} steps...\n")
    
    result = pixel_attack(
        image, model, label, TARGET_CLASS,
        max_pixels=MAX_PIXELS_SIMUL, max_steps=MAX_STEPS, step_size=STEP_SIZE, epsilon=EPSILON, rerank_every=RERANK_EVERY, 
    )

    (perturbed_img, true_confs, target_confs, pred_history,
     flip_step, noise_map, early_imp, mid_imp, final_imp) = result

    final_pred = pred_history[-1]
    print(f"\nFinal prediction after {MAX_STEPS} steps: {final_pred}")
    if flip_step:
        print(f"First flipped to {TARGET_CLASS} at step {flip_step}")
    else:
        print(f"Never reached target class {TARGET_CLASS}")

    fig = plt.figure(figsize=(16, 10))

    ax0 = fig.add_subplot(3, 4, 1)
    ax0.imshow(image.squeeze().numpy(), cmap='gray')
    ax0.set_title(f"Original  (true: {label})")
    ax0.axis('off')

    ax1 = fig.add_subplot(3, 4, 2)
    ax1.imshow(perturbed_img.numpy(), cmap='gray')
    ax1.set_title(f"After {MAX_STEPS} steps  →  pred: {final_pred}")
    ax1.axis('off')

    ax2 = fig.add_subplot(3, 4, 3)
    ax2.imshow(noise_map, cmap='bwr', vmin=-EPSILON, vmax=EPSILON)
    ax2.set_title("Cumulative pixel drift\n(red=brighter, blue=darker)")
    ax2.axis('off')

    ax3 = fig.add_subplot(3, 4, 4)
    ax3.imshow(image.squeeze().numpy(), cmap='gray')
    ax3.imshow(abs(noise_map) > 0.01, cmap='Reds', alpha=0.5)
    ax3.set_title("Pixels that moved")
    ax3.axis('off')

    ax4 = fig.add_subplot(3, 4, 5)
    ax4.imshow(early_imp.numpy(), cmap='hot')
    ax4.set_title("Importance — step 1")
    ax4.axis('off')

    ax5 = fig.add_subplot(3, 4, 6)
    ax5.imshow(mid_imp.numpy(), cmap='hot')
    ax5.set_title(f"Importance — step {MAX_STEPS // 2}")
    ax5.axis('off')

    ax6 = fig.add_subplot(3, 4, 7)
    ax6.imshow(final_imp.numpy(), cmap='hot')
    ax6.set_title(f"Importance — step {MAX_STEPS}")
    ax6.axis('off')

    ax7 = fig.add_subplot(3, 4, 8)
    all_importance = torch.stack([early_imp, mid_imp, final_imp]).mean(dim=0)
    ax7.imshow(all_importance.numpy(), cmap='hot')
    ax7.set_title("Average importance")
    ax7.axis('off')

    # Confidence + prediction trace
    ax8 = fig.add_subplot(3, 1, 3)
    xs = range(1, MAX_STEPS + 1)
    ax8.plot(xs, true_confs,   color='steelblue', linewidth=2, label=f'Confidence in {label} (true)')
    ax8.plot(xs, target_confs, color='tomato',    linewidth=2, label=f'Confidence in {TARGET_CLASS} (target)')

    # shade background by what the model is predicting each step
    prev = pred_history[0]
    seg_start = 0
    colors = {label: 'lightblue', TARGET_CLASS: 'lightsalmon'}
    for i, p in enumerate(pred_history):
        if p != prev or i == MAX_STEPS - 1:
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
        f"Full {MAX_STEPS}-step run  —  {label} → target {TARGET_CLASS}  "
        f"(step_size={STEP_SIZE}, epsilon={EPSILON})",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()