import argparse
import os
import sys
import subprocess
from pathlib import Path

def main(args):

    print(f"Eval Script: {args.eval_script}")
    print(f"Checkpoint Directory: {args.checkpoint_directory}")

    # Check if the eval script exists and is executable
    if not (os.path.isfile(args.eval_script) and os.access(args.eval_script, os.X_OK)):
        print(f"Error: {args.eval_script} does not exist or is not executable.")
        sys.exit(1)

    # Find files matching the pattern and iterate through them
    checkpoint_files = Path(args.checkpoint_directory).rglob("model_*.rank_0.pth")
    for file in checkpoint_files:
        print(f"Submitting job for {file}...")
        subprocess.run(["sbatch", args.eval_script, str(file)], check=True)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_script", type=str, help="Path to the eval script. Assumes that the script is able to read the checkpoint as first argument.", required=True
    )
    parser.add_argument(
        "--checkpoint_directory", type=str, help="Directory containing the checkpoint files", required=True
    )
    parser.add_argument(
        "--checkpoint_pattern", type=str, help="Pattern to match the checkpoint files", default="model_*.rank_0.pth"
    )

    return parser

if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))

# example usage: python scripts/eval_train.py --eval_script /home/hk-project-p0021769/hgf_twg7490/Masterproject-plankton-dinov2/eval_with_checkpoint.sh --checkpoint_directory /home/hk-project-p0021769/hgf_twg7490/test_eval_with_checkpoints