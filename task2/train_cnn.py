import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model import CNNModel, count_parameters
from utils import cutmix_data, get_dataloaders
import warnings
from tqdm import tqdm

train_times = 3
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir=f'runs/cnn_{train_times}')
epochs = 100
batch_size = 16


def train():

    trainloader, valloader, _ = get_dataloaders(batch_size, val_split=0.1)

    model = CNNModel().to(device)
    #model.load_state_dict(torch.load('cnn.pth'))
    #dummy_input = torch.zeros((1, 3, 224, 224)).to(device)
    #writer.add_graph(model, dummy_input)
    print(f'Model: CNN, Parameters: {count_parameters(model)}')

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)  # 使用SGD优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    save_path = 'modelfile'
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for i, (inputs, targets) in enumerate(trainloader):

            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)  #CutMix数据增强

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)  # CutMix loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(trainloader):.4f}')
        writer.add_scalar('Loss/train', running_loss / len(trainloader), epoch)

        scheduler.step()

        model.eval()

        correct = 0
        total = 0
        loss_total = 0.0

        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_total += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100. * correct / total
        print(f'\nEpoch [{epoch + 1}], val Loss: {loss_total / len(valloader):.4f}, val Accuracy: {val_acc:.2f}%')

        writer.add_scalar(f'Loss/val', loss_total / len(valloader), epoch)
        writer.add_scalar(f'Accuracy/val', val_acc, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, f'cnn_{train_times}.pth'))

    writer.close()


if __name__ == "__main__":
    train()
