import argparse
from src.model import GraphEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    model = GraphEncoder(in_dim=32, hidden_dim=128)
    print('evaluating checkpoint', args.checkpoint)


if __name__ == '__main__':
    main()
