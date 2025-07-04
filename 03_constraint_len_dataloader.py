from datasets import load_dataset
from torch.utils.data import DataLoader

from mmdp import config
from mmdp.processors import get_image_processor, get_tokenizer
from mmdp.torch_collators import VQACollator
from mmdp.torch_datasets import VQADataset
from mmdp.utils import visualize_padding

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

    train_dataset = VQADataset(
        dataset=train_ds.select(range(total_samples)),
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token_length=config.image_token_length,
    )

    # Use the VQACollator that sets the maximum length to pad each batch
    # Providing max length does two things
    # 1. Removes a sample from the minibatch that is bigger than max legth
    # 2. Pads till the max length
    vqa_collator = VQACollator(tokenizer=tokenizer, max_length=config.lm_max_length)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        collate_fn=vqa_collator,
    )

    # Showcase the batch elements and the shape
    sample_batch = next(iter(train_dataloader))
    for key, value in sample_batch.items():
        if isinstance(value, list):
            print(key, len(value))
        else:
            print(key, value.shape)

    # Visualize the padding
    visualize_padding(
        sample_batch,
        config.lm_max_length,
        title=f"Constraint Padding: Padded to Fixed Max Length ({config.lm_max_length})",
        fname="assets/03.png",
    )
