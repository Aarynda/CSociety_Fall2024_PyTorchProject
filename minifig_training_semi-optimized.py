#new dataset?????

import torch
from torch import nn
import os
import pandas as pd
from torchvision.transforms import v2
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


#new custom dataset loader class extends standard dataloader class
class MinifigDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=v2.Compose([v2.Grayscale(), v2.CenterCrop(202)]), target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
#new dataloaders need specific links to the folder and annotations file
training_data = MinifigDataset(annotations_file = "minifig_data/index.csv", img_dir = "minifig_data/")
test_data = MinifigDataset(annotations_file = "minifig_data/test.csv", img_dir = "minifig_data/")

batch_size = 5

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = True)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


#IMPORTANT STUFF
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        #considerations for large datasets
        # datasets with large/rgb images require a ton of input neurons
        # if you try to use a fully-connected layer, you can easily cap out the RAM available in your device
        # if the program tries to allocate more RAM than there is, it will throw an error, but if you allocate just a bit less than your maximum
        #       RAM, it will run the program, but may causes crashes/issues with the computer
        # If you have experience with virtual machines/environments, that might help issues
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 1, 3),
            nn.Flatten(),
            nn.Linear(40000, 16384),
            nn.ReLU(),
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 38)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#maps model to whatever device you're using
model = NeuralNetwork().to(device)
print(model)

#defines the loss function and optimization model/learning rate for the network
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=7e-3) #1e-3

#training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        
        X, y = X.to(torch.float32).to(device) / 256, y.to(device) - 1
        #print(type(X))
        #X = [n / 256 for n in X]
        # Compute prediction error
        pred = model(X)
        #print(torch.Size(pred))
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #not necessary, just lets you know that yes, the program is running
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#testing function for measuring current accuracy
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(torch.float32).to(device) / 256, y.to(device) - 1
            
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#how many generations to run the train/test cycle
epochs = 30

#manages the test/train loop
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#save the final model output
torch.save(model.state_dict(), "minifig_model.pth")
print("Saved PyTorch Model State to minifig_model.pth")
