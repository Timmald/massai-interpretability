import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
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
# Set random seed for reproducibility
torch.manual_seed(42)


# Takes an image, a model, and a target class
# (probably integer) and outputs noise (same shape as the image)

#Right now its learning the noise values to learn how far off 
# it is from the target class, but we are trying to maximise the loss 
# that looks pretty simalr to the loss function that they are using that
#  can point to one specfic score and say it is trying to raise the value of
#  that one dimetion/specific score 
#we are tyring to get a specfic score to be wrong and have the rest be null. We can 
# use a cross entropy loss function 
#is it outputing a properility? what is the softmax 
#if it is not outputting a 0 or a 1 then we can use a cross entropy loss function to get the model to be wrong about the target class and right about the rest of the classes.
#if it is already normalising then we need to change it 
#instead of maximising diff between target class we should minimise the 

#T


def generate_noise(image, model: nn.Module, target_class: int):
    model.eval()
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)
    original_shape = image.shape
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.clone().detach().requires_grad_(True)
    output = model(image)
    loss = F.cross_entropy(output, torch.tensor([target_class]))
    model.zero_grad()
    loss.backward()
    # Negate sign: minimizing loss pushes model toward target_class
    noise = -image.grad.data
    return noise.view(original_shape)

# Example usage:
if __name__ == "__main__":
    # Load the pretrained model
    model = Net()
    model.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
    
    # Load a sample image from the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    image, _ = test_dataset[1]  # Get the first image and its label
    _,label = test_dataset[0]
    
    target_class = (label) % 10  # Choose a target class different from the true label
    noise = generate_noise(image, model, target_class)
    
    # Visualize the original image and the generated noise
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Noise")
    plt.imshow(noise.squeeze().numpy(), cmap='berlin')
    plt.axis('off')
    
    plt.show()
           
    