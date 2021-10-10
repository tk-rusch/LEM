from torch import nn, optim
from torch.utils.data import DataLoader
import network
import torch
import data
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--epochs', type=int, default=500,
                    help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00211,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.2371,
                    help='parameter $Delta t$ of LEM')
parser.add_argument('--seed', type=int, default=5544,
                    help='random seed')

args = parser.parse_args()
print(args)

ninp = 2
nout = 1

batch_test = 100

train_dataset, test_dataset, valid_dataset = data.heart_rate()
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)
testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_test)
validloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_test)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model = network.LEM(ninp, args.nhid, nout, args.dt).to(args.device)

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
objective_test = nn.MSELoss(reduction='sum')

def test(dataloader,dataset):
    model.eval()
    loss = 0.
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data = data.permute(1, 0, 2)
            output = model(data.to(args.device)).squeeze(-1)
            loss += objective_test(output, label.to(args.device))
        loss /= len(dataset)
        loss = torch.sqrt(loss)
    return loss.item()

best_eval = 1000000.
for epoch in range(args.epochs):
    model.train()
    for i, (data, label) in enumerate(trainloader):
        optimizer.zero_grad()
        data = data.permute(1, 0, 2)
        output = model(data.to(args.device)).squeeze(-1)
        loss = objective(output, label.to(args.device))
        loss.backward()
        optimizer.step()

    valid_loss = test(validloader, valid_dataset)
    test_loss = test(testloader, test_dataset)
    if (valid_loss < best_eval):
        best_eval = valid_loss
        final_test_loss = test_loss

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/heart_rate_log.txt', 'a')
    f.write('eval loss: ' + str(round(valid_loss, 2)) + '\n')
    f.close()

    if (epoch + 1) == 250:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

f = open('result/heart_rate_log.txt', 'a')
f.write('final test loss: ' + str(round(final_test_loss, 2)) + '\n')
f.close()

