import os
import torch
import torchvision
from pre import PreDataset_CIFAR100, train_transform, test_transform
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='runs/logs_from_scratch')
"""训一次改一次logdir"""

def evaluate(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    return accuracy, loss_total / len(dataloader)

def main():

    # CIFAR-100数据集
    train_set = PreDataset_CIFAR100(root='data', train=True, transform=train_transform, download=True)
    test_set = PreDataset_CIFAR100(root='data', train=False, transform=test_transform, download=True)
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=16)
    testloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=16)

    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_avg = train_loss / len(trainloader)
        test_accuracy, test_loss = evaluate(model, testloader, criterion)

        writer.add_scalar('Loss/train', train_loss_avg, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_avg:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'modelfile/from_scratch_epoch{epoch + 1}.pth')

    writer.close()

if __name__ == "__main__":
    main()
