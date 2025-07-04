from datasets import load_dataset
from torch.utils.data import DataLoader

from mmdp import config
from mmdp.advanced_torch_datasets import ConstantLengthDataset
from mmdp.processors import get_image_processor, get_tokenizer
from mmdp.torch_collators import VQACollator
from mmdp.torch_datasets import VQADataset

# Get the image processor and the text tokenizer
image_processor = get_image_processor(img_size=config.vit_img_size)
tokenizer = get_tokenizer(
    name=config.lm_tokenizer,
    extra_special_tokens=config.vlm_extra_tokens,
    chat_template=config.lm_chat_template,
)

# Load a slice of the total dataset
train_ds = load_dataset(
    path=config.train_dataset_path, name=config.train_dataset_name, split="train"
)
total_samples = min(len(train_ds), config.data_cutoff_idx)

val_size = int(total_samples * config.val_ratio)
train_size = total_samples - val_size


train_dataset = VQADataset(
    dataset=train_ds.select(range(train_size)),
    tokenizer=tokenizer,
    image_processor=image_processor,
    mp_image_token_length=config.image_token_length,
)

train_dataset = ConstantLengthDataset(
    train_dataset,
    infinite=False,
    max_sample_length=config.max_sample_length,
    seq_length=config.lm_max_length,
    num_of_sequences=config.batch_size * 64,
    queue_size=config.batch_size * 64 * 2,
    max_images_per_example=config.max_images_per_example,
    max_images_per_knapsack=config.max_images_per_knapsack,
)
val_dataset = VQADataset(
    train_ds.select(range(train_size, total_samples)),
    tokenizer,
    image_processor,
    config.image_token_length,
)

# Create collators
vqa_collator = VQACollator(tokenizer, config.lm_max_length)


# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,  # =per device BS in DDP
    collate_fn=vqa_collator,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    collate_fn=vqa_collator,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
