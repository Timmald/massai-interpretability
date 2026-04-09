import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


pretrained_model = "data/lenet_mnist_model.pth"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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


def predict(model, img_2d):
    """img_2d: (28, 28) tensor"""
    with torch.no_grad():
        out = model(img_2d.unsqueeze(0).unsqueeze(0))  # (1,1,28,28)
        return out.argmax(dim=1).item(), F.softmax(out, dim=1)


def greedy_pixel_attack(image_2d, model, true_label):
    """
    image_2d: (28, 28) tensor
    Greedily flip pixels (ranked by gradient magnitude) to their extremes.
    Tracks confidence in true class after each flip. Stops when prediction changes.
    """
    model.eval()

    # Compute gradients on the original image
    img = image_2d.unsqueeze(0).unsqueeze(0).clone().detach().requires_grad_(True)  # (1,1,28,28)
    loss = F.cross_entropy(model(img), torch.tensor([true_label]))
    model.zero_grad()
    loss.backward()
    grads = img.grad.squeeze()  # (28,28)

    # Rank pixels by gradient magnitude (most impactful first)
    flat_grads = grads.view(-1)
    ranked = torch.argsort(flat_grads.abs(), descending=True)

    perturbed = image_2d.clone()  # (28,28), we'll modify in-place
    noise_map = torch.zeros(28 * 28)

    confidences = []
    flip_k = None
    flip_pred = None

    for k in range(len(ranked)):
        idx = ranked[k].item()
        sign = flat_grads[idx].sign().item()

        row, col = idx // 28, idx % 28
        perturbed[row, col] = 1.0 if sign >= 0 else 0.0
        noise_map[idx] = sign

        pred, probs = predict(model, perturbed)
        conf = probs[0, true_label].item()
        confidences.append(conf)

        if pred != true_label:
            flip_k = k + 1
            flip_pred = pred
            break

    return perturbed, confidences, flip_k, flip_pred, noise_map.view(28, 28)


if __name__ == "__main__":
    model = Net()
    model.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    image, label = test_dataset[0]
    image_2d = image.squeeze()  # (28,28)

    orig_pred, _ = predict(model, image_2d)
    print(f"True label: {label}, original prediction: {orig_pred}")

    perturbed, confidences, flip_k, flip_pred, noise_map = greedy_pixel_attack(image_2d, model, label)

    print(f"Fooled at k={flip_k} pixels  →  predicted: {flip_pred}")

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(perturbed.numpy(), cmap='gray')
    ax1.set_title(f"Fooled at k={flip_k}  →  pred: {flip_pred}  (true: {label})")
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(noise_map.numpy(), cmap='bwr', vmin=-1, vmax=1)
    ax2.set_title(f"Perturbation ({flip_k} pixels changed)")
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(range(1, len(confidences) + 1), confidences, color='steelblue')
    if flip_k is not None:
        ax3.axvline(x=flip_k, color='red', linestyle='--', label=f'Flips at k={flip_k}')
        ax3.legend()
    ax3.set_xlabel("k (pixels changed)")
    ax3.set_ylabel(f"Model confidence in true class ({label})")
    ax3.set_title("Confidence vs pixels changed — red line = flip point")
    ax3.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
