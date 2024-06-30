import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pre import PreDataset_CIFAR100,test_transform
from model import SimCLRStage2

import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, dataloader):

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    top5_correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            _, top5_pred = outputs.topk(5, 1, True, True)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            top5_correct += sum([1 if targets[i].item() in top5_pred[i] else 0 for i in range(targets.size(0))])

    accuracy = 100. * correct / total
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    top5_accuracy = 100. * top5_correct / total

    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')
    print(f'Test Top-5 Accuracy: {top5_accuracy:.2f}%')

    plt.figure(figsize=(20, 17))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('preds')
    plt.ylabel('targets')
    plt.title(f'{model.__class__.__name__}')
    plt.show()
def main():

    test_set = PreDataset_CIFAR100(root='data', train=False, transform=test_transform, download=True)
    testloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=16)

    model1 = SimCLRStage2(num_class=100).to(device)
    model1.load_state_dict(torch.load('u+lcp10_50.pth'))

    model2 = SimCLRStage2(num_class=100).to(device)
    model2.load_state_dict(torch.load('u+lcp10_100.pth'))

    model3 = SimCLRStage2(num_class=100).to(device)
    model3.load_state_dict(torch.load('u+lcp10_300.pth'))

    model4 = model = torchvision.models.resnet18(pretrained=False)
    model4.fc = nn.Linear(model.fc.in_features, 100)
    model4 = model.to(device)
    model4.load_state_dict(torch.load('imagenet+lcp10.pth'))

    model5 = model = torchvision.models.resnet18(pretrained=False)
    model5.fc = nn.Linear(model.fc.in_features, 100)
    model5 = model.to(device)
    model5.load_state_dict(torch.load('0train_10.pth'))

    evaluate(model1, testloader)
    print("\n")
    evaluate(model2, testloader)
    print("\n")
    evaluate(model3, testloader)
    print("\n")
    evaluate(model4, testloader)
    print("\n")
    evaluate(model5, testloader)

if __name__ == "__main__":
    main()
