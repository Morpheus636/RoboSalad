import os

from aitextgen import aitextgen
from aitextgen.TokenDataset import TokenDataset


model_type = "EleutherAI/gpt-neo-125M"  # type of GPT-2, in this case the smallest version of EleutherAI's gpt-neo
# set save path for the model to path where it gets loaded from when using it
parent_dir = os.path.join(os.getcwd(), os.pardir)
checkpoint_dir = os.path.abspath(
    os.path.join(parent_dir, "src/community_manager/qotd/engine/checkpoint")
)
print(f"Saving checkpoints at: {checkpoint_dir}")

ai = aitextgen(model=model_type, to_gpu=True)
dataset = TokenDataset(file_path="data.txt")
# training settings for a GPU with 8GB VRAM, if you get OOM errors try setting batch_size=1
ai.train(
    dataset,
    batch_size=2,
    num_steps=300,
    save_every=100,
    generate_every=100,
    output_dir=checkpoint_dir,
)