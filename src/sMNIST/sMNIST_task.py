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
parser.add_argument('--batch', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0018,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.21,
                    help='parameter $Delta t$ of LEM')
parser.add_argument('--grad_norm', type=float, default=1.,
                    help='max norm for gradient clipping')
parser.add_argument('--seed', type=int, default=5544,
                    help='random seed')

args = parser.parse_args()
print(args)


ninp = 1
nout = 10
bs_test = 1000

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model = network.LEM(ninp, args.nhid, nout, args.dt).to(args.device)

## get data
train_loader, valid_loader, test_loader = utils.get_data(args.batch,bs_test)

## define the loss
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            ## Reshape images for sequence learning:
            images = images.reshape(images.size(0), 1, 784)
            images = images.permute(2, 0, 1)

            output = model(images.to(args.device))
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred).to(args.device)).sum()
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()

best_eval = 0.
for epoch in range(args.epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        ## Reshape images for sequence learning:
        images = images.reshape(images.size(0), 1, 784)
        images = images.permute(2, 0, 1)

        optimizer.zero_grad()
        output = model(images.to(args.device))
        loss = objective(output, labels.to(args.device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/sMNIST_log.txt', 'a')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch + 1) % 100 == 0:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

f = open('result/sMNIST_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()
