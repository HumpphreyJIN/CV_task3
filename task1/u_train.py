import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import SimCLRStage1, SimCLRStage2
from pre import PreDataset_CIFAR100, train_transform, test_transform

pth_epoch = 300 #加载训练了多少轮的权重
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=f'runs/logs_task_u_eval_01_{pth_epoch}')
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

    # 加载预训练的ResNet-18模型
    simclr_stage1 = SimCLRStage1().to(device)
    checkpoint = torch.load(f'modelfile/model_stage1_epoch{pth_epoch}.pth')
    simclr_stage1.load_state_dict(checkpoint)

    # 创建线性分类器
    model = SimCLRStage2(num_class=100).to(device)
    model.f.load_state_dict(simclr_stage1.f.state_dict())  # 加载预训练的encoder

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)

    # 训练线性分类器
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

        # 评估模型
        train_loss_avg = train_loss / len(trainloader)
        test_accuracy, test_loss = evaluate(model, testloader, criterion)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss_avg, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss_avg:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%')

        # 保存模型权重
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'modelfile/linear_classifier_epoch{epoch + 1}_{pth_epoch}.pth')

    writer.close()

if __name__ == "__main__":
    main()

