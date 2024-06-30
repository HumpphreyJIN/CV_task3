import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import SimCLRStage1,Loss
from pre import PreDataset_CIFAR10,train_transform

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
save_path = "modelfile"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(log_dir='runs/logs_task_u_train_01')
os.makedirs(save_path, exist_ok=True)
epochs = 300
batch_size = 64


def train_simclr_1(dataloader):

    model = SimCLRStage1().to(device)
    lossLR = Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch, (imgL, imgR, labels) in enumerate(dataloader):
            imgL, imgR, labels = imgL.to(device), imgR.to(device), labels.to(device)

            _, pre_L = model(imgL)
            _, pre_R = model(imgR)

            loss = lossLR(pre_L, pre_R, batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

        print(f"epoch{epoch+1}/{epochs} loss:", epoch_loss / len(dataloader))
        writer.add_scalar(tag='Loss/train',scalar_value= (epoch_loss / len(dataloader)), global_step=epoch)

        if (epoch+1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_stage1_epoch' + str(epoch+1) + '.pth'))


def main():

    train_set = PreDataset_CIFAR10(root='data', train=True, transform=train_transform, download=True)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)
    train_simclr_1(trainloader)
    writer.close()

if __name__ == "__main__":
    main()
