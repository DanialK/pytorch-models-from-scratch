import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0,0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0,0))
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def train(model: LeNet,
          data: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          device: torch.device) -> None:

    model.train()

    train_loss = 0

    for (image, target) in data:
        image, target = image.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, target)

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

    train_loss /= len(data.dataset)

    print('Train set: Average loss: {:.4f}'.format(train_loss))


def evaluate(model: LeNet,
             data: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device: torch.device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, target in data:
            image, target = image.float().to(device), target.to(device)
            output = model(image)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data.dataset),
        100. * correct / len(data.dataset)))


if __name__ == "__main__":
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 1024

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Pad(2)  # make 28 x 28 -> 32 x 32
    ])

    mnist_train = MNIST(root=".data", train=True, download=True, transform=transform)
    mnist_train_data_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

    mnist_test = MNIST(root=".data", train=False, download=True, transform=transform)
    mnist_test_data_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train(model, mnist_train_data_loader, optimizer, criterion, device)
        evaluate(model, mnist_test_data_loader, criterion, device)
