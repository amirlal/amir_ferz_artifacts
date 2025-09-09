# model_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from activation import LegacyTransmute
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False
)

class SimpleNet(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.act1 = activation_fn
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act1(self.fc1(x))
        return self.fc2(x)

def train(model, optimizer, criterion):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

def run_experiment(name, activation_fn):
    print(f"\nRunning: {name}")
    model = SimpleNet(activation_fn).to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    train(model, optimizer, criterion)
    acc = test(model)
    return acc

if __name__ == "__main__":
    results = {}
    results['ReLU'] = run_experiment("ReLU", nn.ReLU())
    results['GELU'] = run_experiment("GELU", nn.GELU())
    results['LegacyTransmute'] = run_experiment("LegacyTransmute", LegacyTransmute(alpha=1.0, beta=0.5, gamma=0.7))

    plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy")
    plt.title("Activation Function Comparison")
    plt.savefig("activation_comparison.png")
    print("Saved plot to activation_comparison.png")
