import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ── change these ──────────────────────────
pretrained_model = "data/lenet_mnist_model.pth"
IMAGE_INDEX      = 0
TARGET_CLASS     = 3
MAX_PIXELS       = 500
RERANK_EVERY     = 10
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
    return -img.grad.squeeze().detach()


def greedy_pixel_attack(image, model, true_label, target_class, max_pixels=500, rerank_every=10):
    """
    Targeted greedy pixel attack with periodic re-ranking.
    Every `rerank_every` steps, recompute gradients from the current
    perturbed image so the attack adapts as the image changes.
    Stops when model predicts target_class.
    """
    model.eval()

    perturbed        = image.clone()
    flat_perturbed   = perturbed.view(-1)
    flat_noise       = torch.zeros(784)
    already_flipped  = set()

    true_confidences   = []
    target_confidences = []
    flip_k             = None
    flip_pred          = None
    flat_grads         = None
    rank_queue         = []

    for k in range(max_pixels):

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
            break

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

    if initial_pred != label:
        print("Model already wrong — try a different IMAGE_INDEX")
    elif label == TARGET_CLASS:
        print("TARGET_CLASS is same as true label — pick a different target")
    else:
        result = greedy_pixel_attack(
            image, model, label, TARGET_CLASS,
            max_pixels=MAX_PIXELS, rerank_every=RERANK_EVERY
        )
        perturbed_img, true_confs, target_confs, flip_k, flip_pred, noise_map = result

        if flip_k is None:
            print(f"Did not reach target class {TARGET_CLASS} within {MAX_PIXELS} pixels")
            print("Try: increase MAX_PIXELS, decrease RERANK_EVERY, or pick a closer target class")
        else:
            print(f"Success — fooled at k={flip_k}  →  model now says: {flip_pred}")

            fig = plt.figure(figsize=(14, 8))

            # Original
            ax0 = fig.add_subplot(2, 3, 1)
            ax0.imshow(image.squeeze().numpy(), cmap='gray')
            ax0.set_title(f"Original  (true: {label})")
            ax0.axis('off')

            # Perturbed
            ax1 = fig.add_subplot(2, 3, 2)
            ax1.imshow(perturbed_img.squeeze().numpy(), cmap='gray')
            ax1.set_title(f"After attack  →  pred: {flip_pred}")
            ax1.axis('off')

            # Noise map
            ax2 = fig.add_subplot(2, 3, 3)
            ax2.imshow(noise_map, cmap='bwr', vmin=-1, vmax=1)
            ax2.set_title(f"{flip_k} pixels changed\n(red=pushed up, blue=pushed down)")
            ax2.axis('off')

            # Importance map toward target
            grads = get_pixel_gradients(image, model, TARGET_CLASS)
            ax3 = fig.add_subplot(2, 3, 4)
            im = ax3.imshow(grads.abs().numpy(), cmap='hot')
            ax3.set_title(f"Pixel importance toward {TARGET_CLASS}\n(bright = helps reach target)")
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, fraction=0.046)

            # Attacked pixels overlay
            ax4 = fig.add_subplot(2, 3, 5)
            ax4.imshow(image.squeeze().numpy(), cmap='gray')
            attacked_mask = (noise_map != 0).astype(float)
            ax4.imshow(attacked_mask, cmap='Reds', alpha=0.6)
            ax4.set_title(f"Pixels touched  ({flip_k} total)")
            ax4.axis('off')

            # Dual confidence curve
            ax5 = fig.add_subplot(2, 3, 6)
            xs = range(1, len(true_confs) + 1)
            ax5.plot(xs, true_confs,   color='steelblue', linewidth=2, label=f'Confidence in {label} (true)')
            ax5.plot(xs, target_confs, color='tomato',    linewidth=2, label=f'Confidence in {TARGET_CLASS} (target)')
            ax5.axvline(x=flip_k, color='black', linestyle='--', label=f'Flips at k={flip_k}')
            ax5.legend(fontsize=8)
            ax5.set_xlabel("Pixels changed (k)")
            ax5.set_ylabel("Model confidence")
            ax5.set_title("True class drops, target class rises")
            ax5.set_ylim(0, 1)

            plt.suptitle(
                f"Targeted Greedy Attack  —  {label} → {TARGET_CLASS}  in {flip_k} pixels",
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()
            plt.show()