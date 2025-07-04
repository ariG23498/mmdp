from torchvision import transforms
from transformers import AutoTokenizer


def get_tokenizer(name, extra_special_tokens=None, chat_template=None):
    tokenizer_init_kwargs = {"use_fast": True}
    if extra_special_tokens is not None:
        tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
    if chat_template is not None:
        tokenizer_init_kwargs["chat_template"] = chat_template
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        **tokenizer_init_kwargs,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_image_processor(img_size):
    return transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )
