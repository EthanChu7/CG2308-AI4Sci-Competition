import argparse

def get_config_task1():

    parser = argparse.ArgumentParser(description='task1')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    return args


def get_config_task2():

    parser = argparse.ArgumentParser(description='task2')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    return args

def get_config_baseline_gnn_task1():
    parser = argparse.ArgumentParser(description='baseline_gnn_task1')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='gcn', choices=['gcn', 'gat'])
    args = parser.parse_args()

    return args


def get_config_baseline_gnn_task2():
    parser = argparse.ArgumentParser(description='baseline_gnn_task1')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='gcn', choices=['gcn', 'gat'])
    args = parser.parse_args()

    return args