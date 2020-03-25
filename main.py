import sys

import argparse
from train.training_main import train_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='None')
    parser.add_argument('-M', '--mode', type=str, default='train')
    parser.add_argument('-d', '--dataset', type=str, default='chengdu_sample')
    parser.add_argument('-i', '--identify', type=str, default='')
    parser.add_argument('-D', '--device', type=str, default="cuda:0")

    parser.add_argument('-o', '--optim', type=str, default="Adam")
    parser.add_argument('-C', '--lossinside', type=bool, default=False)
    parser.add_argument('-c', '--loss', type=str, default="mse")
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-l', '--lr', type=float, default=0.001)

    args = parser.parse_args()
    print(args.model)
    if args.model == "deeptravel":
        from models.deeptravel.DeepTravel.main import main
        main()
    else:
        print(args.dataset)
        train_main(args)
    sys.exit(0)
