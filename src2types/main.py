from utils import tab_printer
from hybgnn import HybGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a HybGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = HybGNNTrainer(args)
    if args.load_path:
        trainer.load()
    else:
        trainer.fit()
    trainer.score()
    if args.save_path:
        trainer.save()

if __name__ == "__main__":
    main()
