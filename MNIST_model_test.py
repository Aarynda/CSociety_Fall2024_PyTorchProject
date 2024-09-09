#MNIST_model_test

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 676),
            nn.ReLU(),
            nn.Linear(676, 576),
            nn.ReLU(),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("MNIST_model.pth"))

classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]

NUM_TESTS = 1000

mistakes = 0

with torch.no_grad():
    for i in range(NUM_TESTS):
        x, y = test_data[i][0], test_data[i][1]
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        if(predicted != actual):
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
            mistakes = mistakes + 1

print(f"Accuracy: {NUM_TESTS - mistakes} / {NUM_TESTS}; {(NUM_TESTS - mistakes) / NUM_TESTS}%")