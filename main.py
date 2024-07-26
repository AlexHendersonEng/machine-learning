from util import CustomMNIST
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Select train or test mode
train = True

# Get GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = CustomMNIST('./data', transform=transform, train=False, download=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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
        return F.softmax(self.fc3(x))


# Instantiate model, loss function and optimiser
model = DigitClassifier().to(device)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=128, gamma=0.9)

# Create tensorboard summary writer
writer = SummaryWriter()

# Training loop
if train:
    for epoch in range(3):
        for batch_idx, data in enumerate(train_dataloader):
            # Get images and labels
            images, labels = data[0].to(device), data[1].to(device)

            # One hot encoding
            one_hot = torch.zeros(labels.size(0), 10).to(device)
            for row in range(one_hot.size(0)):
                one_hot[row, labels[row].item()] = 1

            # Zero gradients for every batch
            optim.zero_grad()

            # Make predictions for batch
            preds = model(images)

            # Compute loss and gradients
            loss = loss_fn(preds, one_hot)
            loss.backward()

            # Adjust model parameters
            optim.step()

            # Update learning rate
            scheduler.step()

            # Write loss statistic to tensorboard
            writer.add_scalar('Loss', loss.item(), epoch * len(train_dataloader) + batch_idx)

            # Print loss to command line
            print(f"Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item()}," +
                  f" Learning rate: {optim.param_groups[0]['lr']}")

    # Save model weights
    torch.save(model.state_dict(), './model.pth')

    # Close summary writer
    writer.close()
else:
    model.load_state_dict(torch.load('./model.pth'))

# Testing loop
model.eval()
with torch.no_grad():
    num_correct = 0
    for data in test_dataloader:
        # Get images and labels
        images, labels = data[0].to(device), data[1].to(device)

        # Make predictions
        preds = model(images)

        # Get number of correct predictions
        num_correct += torch.sum(torch.argmax(preds, dim=1) == labels).item()

# Print accuracy
acc = num_correct / len(test_dataset)
print(f'Accuracy: {acc}')

# Make some predictions
for i in range(9):
    # Make prediction
    image, _ = test_dataset[i]
    image = image.to(device)
    preds = model(image)
    pred = torch.argmax(preds).item()

    # Configure plot
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(image.cpu()[0])
    ax.set_title(f'Prediction: {pred}')
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)

# Show plot
plt.show()








