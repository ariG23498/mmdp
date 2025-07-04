from datasets import load_dataset
from torch.utils.data import DataLoader

from mmdp import config
from mmdp.advanced_torch_datasets import NaiveConstantLengthDataset
from mmdp.processors import get_image_processor, get_tokenizer
from mmdp.torch_collators import VQACollator
from mmdp.torch_datasets import VQADataset
from mmdp.utils import visualize_knapsack_packing

if __name__ == "__main__":
    # Get the image processor and the text tokenizer
    image_processor = get_image_processor(img_size=config.vit_img_size)
    tokenizer = get_tokenizer(
        name=config.lm_tokenizer,
        extra_special_tokens=config.vlm_extra_tokens,
        chat_template=config.lm_chat_template,
    )

    # Load the dataset from Hugging Face
    train_ds = load_dataset(
        path=config.train_dataset_path, name=config.train_dataset_name, split="train"
    )
    total_samples = min(len(train_ds), config.data_cutoff_idx)
    val_size = int(total_samples * config.val_ratio)
    train_size = total_samples - val_size

    # Wrap the dataset into `VQADataset` that processes the images into torch tensors
    # and texts into tokenized encodings
    train_dataset = VQADataset(
        dataset=train_ds.select(range(train_size)),
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token_length=config.image_token_length,
    )

    # Wrap the map styled dataset into an iterable dataset which uses knapsack
    # to pack each data point into the maximum length provided
    packed_ds = NaiveConstantLengthDataset(
        dataset=train_dataset,
        seq_length=config.lm_max_length,
        num_sequences=config.num_sequences,
        max_sample_length=config.max_sample_length,
        max_images_per_example=config.max_images_per_example,
        max_images_per_knapsack=config.max_images_per_knapsack,
        delta=5,
        infinite=False,
    )

    vqa_collator = VQACollator(tokenizer, config.lm_max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # =per device BS in DDP
        collate_fn=vqa_collator,
        pin_memory=True,
        drop_last=True,
    )

    sample_batch = next(iter(train_dataloader))
    for key, value in sample_batch.items():
        if isinstance(value, list):
            print(key, len(value))
        else:
            print(key, value.shape)

    # Visualize the knapsack packing
    visualize_knapsack_packing(
        sample_batch,
        seq_length=config.lm_max_length,
        title=f"Knapsack Packing: Samples Packed into Fixed Length ({config.lm_max_length})",
        fname="04.png",
    )
