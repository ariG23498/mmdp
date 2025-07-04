from datasets import load_dataset
from torch.utils.data import DataLoader
from mmdp import config
from mmdp.torch_datasets import VQADataset
from mmdp.advanced_torch_datasets import NaiveConstantLengthDataset
from mmdp.torch_collators import VQACollator
from mmdp.processors import get_image_processor, get_tokenizer

if __name__ == "__main__":

    # Get the image processor and the text tokenizer
    image_processor = get_image_processor(img_size=config.vit_img_size)
    tokenizer = get_tokenizer(
        name=config.lm_tokenizer,
        extra_special_tokens=config.vlm_extra_tokens,
        chat_template=config.lm_chat_template,
    )

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
        image_token_length=config.image_token_length,
    )
    packed_ds = NaiveConstantLengthDataset(
        train_dataset,
        seq_length=1024,
        num_sequences=1024,
        max_sample_length=1024,
    )


    vqa_collator = VQACollator(tokenizer, config.lm_max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,  # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )


    sample_batch = next(iter(train_dataloader))
    for key, value in sample_batch.items():
        print(key, value.shape)
