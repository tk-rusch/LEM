from torch import nn, optim, Tensor
import torch
import network
import torch.nn.utils
import data
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=16,
                    help='hidden size')
parser.add_argument('--epochs', type=int, default=400,
                    help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00904,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')

args = parser.parse_args()
print(args)

ninp = 1
nout = 1

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

train_x, train_y = data.get_data(128)
valid_x, valid_y = data.get_data(128)
test_x, test_y = data.get_data(1024)
print('Finished generating data')

## Train data:
train_dataset = TensorDataset(Tensor(train_x).float(), Tensor(train_y))
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)

## Valid data
valid_dataset = TensorDataset(Tensor(valid_x).float(), Tensor(valid_y).float())
validloader = DataLoader(valid_dataset, shuffle=False, batch_size=128)

## Test data
test_dataset = TensorDataset(Tensor(test_x).float(), Tensor(test_y).float())
testloader = DataLoader(test_dataset, shuffle=False, batch_size=128)


model = network.LEM(ninp, args.nhid, nout).to(args.device)

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def eval(dataloader):
    model.eval()
    with torch.no_grad():
        for x,y in dataloader:
            y = y.permute(1, 0, 2)
            x = x.permute(1, 0, 2)
            out = model(x.to(args.device))
            loss = torch.sqrt(objective(out,y.to(args.device))).item()
    return loss

best_loss = 10000
for epoch in range(args.epochs):
    model.train()
    for x,y in trainloader:
        y = y.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        optimizer.zero_grad()
        out = model(x.to(args.device))
        loss = objective(out, y.to(args.device))
        loss.backward()
        optimizer.step()
    valid_loss = eval(validloader)
    test_loss = eval(testloader)
    if (valid_loss < best_loss):
        best_loss = valid_loss
        final_test_loss = test_loss

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/FitzHughNagumo_log.txt', 'a')
    f.write('eval loss: ' + str(valid_loss) + '\n')
    f.close()

f = open('result/FitzHughNagumo_log.txt', 'a')
f.write('final test loss: ' + str(final_test_loss) + '\n')
f.close()

