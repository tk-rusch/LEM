from torch import nn, optim
import torch
import torch.nn.utils
import os
import data
import network
from pathlib import Path
from google12_data_loader import GCommandLoader
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--epochs', type=int, default=60,
                    help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=100,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00089,
                    help='learning rate')
parser.add_argument('--drop', type=float, default=0.03,
                    help='input dropout rate')
parser.add_argument('--seed', type=int, default=5544,
                    help='random seed')

args = parser.parse_args()
print(args)

ninp = 80
nout = 12

## get data
data.google12_v2()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model = network.LEM(ninp,args.nhid,nout,drop=args.drop).to(args.device)

cwd = os.getcwd()
train_dataset = GCommandLoader(cwd+'/../../data/google_speech_command/processed/train', window_size=.02)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch, shuffle=True,
    num_workers=12, pin_memory='cpu', sampler=None)

valid_dataset = GCommandLoader(cwd+'/../../data/google_speech_command/processed/valid', window_size=.02)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch, shuffle=None,
    num_workers=12, pin_memory='cpu', sampler=None)

test_dataset = GCommandLoader(cwd+'/../../data/google_speech_command/processed/test', window_size=.02)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch, shuffle=None,
    num_workers=12, pin_memory='cpu', sampler=None)

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test(data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.squeeze(1).permute(2,0,1)

            output = model(data.to(args.device))
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred).to(args.device)).sum()
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

best_eval = 0.
for epoch in range(args.epochs):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data = data.squeeze(1).permute(2,0,1)
        optimizer.zero_grad()
        output = model(data.to(args.device))
        loss = objective(output, target.to(args.device))
        loss.backward()
        optimizer.step()

    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    if (valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open('result/google12_log.txt', 'a')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch + 1) == 50:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

f = open('result/google12_log.txt', 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()


