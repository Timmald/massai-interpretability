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
MAX_STEPS        = 500
STEP_SIZE        = 0.005
EPSILON          = 0.3
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


def continuous_targeted_attack(image, model, true_label, target_class,
                                max_steps=500, step_size=0.01, epsilon=0.3):
    model.eval()

    original  = image.clone()
    perturbed = image.clone().unsqueeze(0)

    true_confidences   = []
    target_confidences = []
    importance_history = []
    flip_step          = None
    flip_pred          = None

    for step in range(max_steps):
        perturbed = perturbed.detach().requires_grad_(True)

        loss = F.cross_entropy(model(perturbed), torch.tensor([target_class]))
        model.zero_grad()
        loss.backward()

        importance = perturbed.grad.abs().squeeze().detach().clone()
        importance_history.append(importance)

        with torch.no_grad():
            perturbed = perturbed - step_size * perturbed.grad.sign()
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

        if pred == target_class and flip_step is None:
            flip_step = step + 1
            flip_pred = pred
            break

    # squeeze both to (28, 28) before subtracting
    noise_map = (perturbed.squeeze() - original.squeeze()).detach().numpy()
    final_img = perturbed.squeeze().detach()

    early_importance = importance_history[0]
    mid_importance   = importance_history[len(importance_history) // 2]
    final_importance = importance_history[-1]

    return (final_img, true_confidences, target_confidences,
            flip_step, flip_pred, noise_map,
            early_importance, mid_importance, final_importance)


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
        print("TARGET_CLASS same as true label — pick a different target")
    else:
        result = continuous_targeted_attack(
            image, model, label, TARGET_CLASS,
            max_steps=MAX_STEPS, step_size=STEP_SIZE, epsilon=EPSILON
        )
        (perturbed_img, true_confs, target_confs,
         flip_step, flip_pred, noise_map,
         early_imp, mid_imp, final_imp) = result

        if flip_step is None:
            print(f"Did not reach target {TARGET_CLASS} within {MAX_STEPS} steps")
            print("Try: increase MAX_STEPS, increase STEP_SIZE, or increase EPSILON")
        else:
            print(f"Success — flipped at step={flip_step}  →  model now says: {flip_pred}")

            fig = plt.figure(figsize=(16, 10))

            ax0 = fig.add_subplot(3, 4, 1)
            ax0.imshow(image.squeeze().numpy(), cmap='gray')
            ax0.set_title(f"Original  (true: {label})")
            ax0.axis('off')

            ax1 = fig.add_subplot(3, 4, 2)
            ax1.imshow(perturbed_img.numpy(), cmap='gray')
            ax1.set_title(f"After attack  →  pred: {flip_pred}")
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
            ax5.set_title(f"Importance — step {flip_step // 2}")
            ax5.axis('off')

            ax6 = fig.add_subplot(3, 4, 7)
            ax6.imshow(final_imp.numpy(), cmap='hot')
            ax6.set_title(f"Importance — step {flip_step}")
            ax6.axis('off')

            ax7 = fig.add_subplot(3, 4, 8)
            all_importance = torch.stack([early_imp, mid_imp, final_imp]).mean(dim=0)
            ax7.imshow(all_importance.numpy(), cmap='hot')
            ax7.set_title("Average importance\nacross all steps")
            ax7.axis('off')

            ax8 = fig.add_subplot(3, 1, 3)
            xs = range(1, len(true_confs) + 1)
            ax8.plot(xs, true_confs,   color='steelblue', linewidth=2, label=f'Confidence in {label} (true)')
            ax8.plot(xs, target_confs, color='tomato',    linewidth=2, label=f'Confidence in {TARGET_CLASS} (target)')
            ax8.axvline(x=flip_step, color='black', linestyle='--', label=f'Flips at step={flip_step}')
            ax8.legend(fontsize=9)
            ax8.set_xlabel("Step")
            ax8.set_ylabel("Model confidence")
            ax8.set_title("Confidence over steps — blue drops, red rises, they cross at the flip")
            ax8.set_ylim(0, 1)

            plt.suptitle(
                f"Continuous Attack  —  {label} → {TARGET_CLASS}  "
                f"in {flip_step} steps  (step_size={STEP_SIZE}, epsilon={EPSILON})",
                fontsize=13, fontweight='bold'
            )
            plt.tight_layout()
            plt.show()