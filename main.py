import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import init
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#载入训练集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.hidden2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.hidden3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.hidden4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)
        init.xavier_normal_(self.output.weight)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.hidden4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.output(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
train_losses, train_accuracies, test_accuracies = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()  
  
    epoch_loss = running_loss / len(train_loader.dataset)  
    train_accuracy = correct_train / total_train  
    train_losses.append(epoch_loss)  
    train_accuracies.append(train_accuracy)  
  
    # 测试  
    model.eval()  
    correct_test = 0  
    total_test = 0  
    with torch.no_grad():  
        for images, labels in test_loader:  
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)  
            _, predicted = torch.max(outputs, 1)  
            total_test += labels.size(0)  
            correct_test += (predicted == labels).sum().item()  
    test_accuracy = correct_test / total_test  
    test_accuracies.append(test_accuracy)  
  
    print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | "          f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")  

torch.save(model.state_dict(), 'mnist_model.pth')
print("模型已成功保存为 mnist_model.pth")
  
# 可视化  
plt.figure(figsize=(10, 5))  
plt.plot(train_losses, label='Train Loss', marker='o')  
plt.plot(train_accuracies, label='Train Accuracy', marker='x')  
plt.plot(test_accuracies, label='Test Accuracy', marker='s')  
plt.title('Training Loss & Accuracy')  
plt.xlabel('Epoch')  
plt.ylabel('Value')  
plt.ylim(0, max(max(train_losses), 1.0) + 0.1)  
plt.grid(True)  
plt.legend()  
plt.show()