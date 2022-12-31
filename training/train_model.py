"""Finetunes an AI model based on data.json."""
import os
import json

from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset

# Define path-related constants
PARENT_DIR = os.path.getcwd()
CHECKPOINT_DIR = os.path.join(PARENT_DIR, "src", "robosalad", "qotd", "engine", "checkpoint")
TRAINING_DIR = os.path.join(PARENT_DIR, "training")
DATA_OUTPUT_FILE = os.path.join(TRAINING_DIR, "data.txt")
DATA_JSON_FILE = os.path.join(TRAINING_DIR, "data.json")

def prep_data():
    """
    """
    with open(DATA_OUTPUT_FILE ) as dataset_formatted:
        with open(DATA_JSON_FILE, "r") as dataset_raw:
            dataset = json.load(dataset_raw)
            for data in dataset:
                dataset_formatted.write(f"<|startoftext|>\n{data}\n<|endoftext|>\n")
    

def finetune_model():
    """
    """
    model_type = "EleutherAI/gpt-neo-125M"  # type of GPT-2, in this case the smallest version of EleutherAI's gpt-neo
    # set save path for the model to path where it gets loaded from when using it
    print(f"Saving checkpoints at: {CHECKPOINT_DIR}")

    ai = aitextgen(model=model_type, to_gpu=True)
    dataset = TokenDataset(file_path=DATA_OUTPUT_FILE)
    # training settings for a GPU with 8GB VRAM, if you get OOM errors try setting batch_size=1
    ai.train(
        dataset,
        batch_size=2,
        num_steps=300,
        save_every=100,
        generate_every=100,
        output_dir=CHECKPOINT_DIR,
    )

def main():
    prep_data()
    finetune_model()

if __name__ == "__main__":
    main()
