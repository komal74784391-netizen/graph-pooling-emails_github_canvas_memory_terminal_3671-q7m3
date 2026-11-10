import argparse
import torch
from src.model import GraphEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    model = GraphEncoder(in_dim=32, hidden_dim=128)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print('training with', args.config)
    for step in range(1000):
        opt.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)
        loss.backward()
        opt.step()


if __name__ == '__main__':
    main()
