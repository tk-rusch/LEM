from torch import nn, optim
import torch
import network
import torch.nn.utils
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--seq_length', type=int, default=10000,
                    help='length of sequences')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--max_iter', type=int, default=30000,
                    help='max number of learning steps')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    help='computing device')
parser.add_argument('--batch', type=int, default=50,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0026,
                    help='learning rate')
parser.add_argument('--dt', type=float, default=0.0242,
                    help='parameter $Delta t$ of LEM')

args = parser.parse_args()
print(args)

def get_batch(T,batch_size):
    add_values = torch.rand(T, batch_size, requires_grad=False)

    # Build the second sequence with one 1 in each half and 0s otherwise
    add_indices = torch.zeros_like(add_values)
    half = int(T / 2)
    for i in range(batch_size):
        first_half = np.random.randint(half)
        second_half = np.random.randint(half, T)
        add_indices[first_half, i] = 1
        add_indices[second_half, i] = 1

    inputs = torch.stack((add_values, add_indices), dim=-1)
    targets = torch.mul(add_values, add_indices).sum(dim=0)
    return inputs, targets

ninp = 2
nout = 1
model = network.LEM(ninp, args.nhid, nout, args.dt).to(args.device)

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test():
    model.eval()
    with torch.no_grad():
        data, label = get_batch(args.seq_length, 1000)
        label = label.unsqueeze(1)
        out = model(data.to(args.device))
        loss = objective(out, label.to(args.device))

    return loss.item()

model.train()
for i in range(args.max_iter):
    data, label = get_batch(args.seq_length,args.batch)
    label = label.unsqueeze(1)
    optimizer.zero_grad()
    out = model(data.to(args.device))
    loss = objective(out, label.to(args.device))
    loss.backward()
    optimizer.step()

    if(i%100==0):
        test_mse = test()
        Path('results').mkdir(parents=True, exist_ok=True)
        f = open('results/adding_log_seqlength_' + str(args.seq_length) + '.txt', 'a')
        f.write(str(round(test_mse, 6)) + '\n')
        f.close()
        model.train()
