from torch import nn, optim
from torch.utils.data import DataLoader
import network
import torch
from pathlib import Path
import data
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=32,
                    help='hidden size')
parser.add_argument('--epochs', type=int, default=50,
                    help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=8,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00609,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.00017,
                    help='parameter $Delta t$ of LEM')
parser.add_argument('--seed', type=int, default=5544,
                    help='random seed')

args = parser.parse_args()
print(args)

ninp = 6
nout = 5

train_dataset, test_dataset, valid_dataset = data.EigenWorms()
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)
testloader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))
validloader = DataLoader(valid_dataset, shuffle=False, batch_size=len(valid_dataset))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model = network.LEM(ninp, args.nhid, nout, args.dt).to(args.device)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, label in dataloader:
            data = data.permute(1, 0, 2)
            output = model(data.to(args.device))
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred).to(args.device)).sum()
    accuracy = 100. * correct / len(dataloader.dataset)
    return accuracy.item()

best_eval = 0.
for epoch in range(args.epochs):
    model.train()
    for i, (data, label) in enumerate(trainloader):
        optimizer.zero_grad()
        data = data.permute(1, 0, 2)
        output = model(data.to(args.device))
        loss = objective(output, label.to(args.device))
        loss.backward()
        optimizer.step()

    valid_acc = test(validloader)
    test_acc = test(testloader)
    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/eigenWorms_log.txt', 'a')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

f = open('result/eigenWorms_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()
