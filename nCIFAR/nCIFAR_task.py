from torch import nn, optim
import torch
import network
import torch.nn.utils
from pathlib import Path
import utils
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=100,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00322,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.44,
                    help='parameter $Delta t$ of LEM')
parser.add_argument('--seed', type=int, default=5544,
                    help='random seed')

args = parser.parse_args()
print(args)

ninp = 96
nout = 10

bs_test = 1000

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model = network.LEM(ninp,args.nhid,nout,args.dt).to(args.device)

train_loader, valid_loader, test_loader = utils.get_data(args.batch,bs_test)

rands = torch.randn(1, 1000 - 32, 96)
rand_train = rands.repeat(args.batch, 1, 1)
rand_test = rands.repeat(bs_test, 1, 1)

## Define the loss
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = torch.cat((images.permute(0,2,1,3).reshape(bs_test,32,96),rand_test),dim=1).permute(1,0,2)
            output = model(images.to(args.device))
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred).to(args.device)).sum()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()

best_eval = 0.
for epoch in range(args.epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = torch.cat((images.permute(0,2,1,3).reshape(args.batch,32,96),rand_train),dim=1).permute(1,0,2)
        optimizer.zero_grad()
        output = model(images.to(args.device))
        loss = objective(output, labels.to(args.device))
        loss.backward()
        optimizer.step()

    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/nCIFAR_log.txt', 'a')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch + 1) % 100 == 0:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

f = open('result/nCIFAR_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()
