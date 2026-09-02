from train import parser, train

if __name__ == "__main__":
    args = parser().parse_args()
    args.algorithm = "matd3"
    train(args)
