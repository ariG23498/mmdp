from datasets import load_dataset
from torch.utils.data import DataLoader

from mmdp import config
from mmdp.processors import get_image_processor, get_tokenizer
from mmdp.torch_collators import NaiveCollator
from mmdp.torch_datasets import VQADataset
from mmdp.utils import visualize_padding

if __name__ == "__main__":
    # Get the image processor and the text tokenizer
    image_processor = get_image_processor(
        img_size=config.vit_img_size
    )  # torchvison transforms compose
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

    # Wrap the dataset into VQADataset which processes the images and texts
    train_dataset = VQADataset(
        dataset=train_ds.select(range(total_samples)),  # select a slice of the dataset
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token_length=config.image_token_length,  # number of image tokens to prepend (64)
    )

    # Use the naive collator to pad each minibatch to the maximum length
    naive_collator = NaiveCollator(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        collate_fn=naive_collator,
    )

    # Showcase the batch elements and the shape
    sample_batch = next(iter(train_dataloader))
    for key, value in sample_batch.items():
        print(key, value.shape)

    # Visualize the padding
    max_len = sample_batch["input_ids"].shape[1]
    visualize_padding(
        sample_batch,
        max_len,
        title="Naive Padding: Padded to Max Length in Batch",
        fname="assets/02.png",
    )
