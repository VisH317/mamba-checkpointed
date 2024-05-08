import argparse
from benchmark.benchmark import finetune

parser = argparse.ArgumentParser(
    prog='DNA model finetune',
    description='fine tunes the attention-augmented DNA model'
)

parser.add_argument("dataset_name", type=str, help="Name of the dataset to finetune the model on")
parser.add_argument("pretrained_path", type=str, help="Path to the pretrained model's state_dict file")
parser.add_argument("--has_lmhead", action="store_true", help="Whether the model has a language model head (default: False)")

# training config
parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default 16)")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate (default 3e-4)")
parser.add_argument("--grad_accum", type=int, default=4, help="Number of steps to run grad accumulation (default 4)")
parser.add_argument("--dropout_p", type=float, default=0.25, help="Dropout probability (default 0.25)")
parser.add_argument("--epochs", type=int, default=5, help="number of epochs")

# no model config params bc just trust i promise

def finetune_from_args(args: argparse.Namespace):
    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "grad_accum_steps": args.grad_accum,
        "dropout_p": args.dropout_p,
        "epoch": args.epochs
    }

    model = finetune(args.dataset_name, args.pretrained_path, train_config=config, has_lmhead=args.has_lmhead)

    print("finetuning complete")

if __name__ == "__main__":
    args = parser.parse_args()
    finetune_from_args(args)
