from util import CustomMNIST
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Calculate mean and std of training dataset
train_dataset = CustomMNIST('./data', transform=transforms.ToTensor(), train=True, download=True)
features, _ = train_dataset[:]


# Create transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(torch.mean(features), torch.std(features))]
)

# Create datasets and dataloaders
train_dataset = CustomMNIST('./data', transform=transform, train=True, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = CustomMNIST('./data', transform=transform, train=False, download=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Digit classifier model
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128, bias=True)
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.fc3 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Instantiate model, loss function and optimiser
model = DigitClassifier()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)

# Training loop
for epoch in range(3):
    for batch_idx, data in enumerate(train_dataloader):
        # Get images and labels
        images, labels = data

        # Zero gradients for every batch
        optim.zero_grad()

        # Make predictions for batch
        preds = model(images)

        # Compute loss and gradients
        loss = loss_fn(preds, labels)
        loss.backward()

        # Adjust model parameters
        optim.step()

        # Print loss to command line
        print(f'Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item()}')

# Test model
model.eval()
test_image, test_label = test_dataset[0]
pred = model(test_image)
print(pred)
plt.imshow(test_image[0])
plt.show()





