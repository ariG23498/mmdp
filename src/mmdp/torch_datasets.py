import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset class for processing text and image data with tokenization."""

    def __init__(self, dataset, tokenizer, image_processor, image_token_length):
        """
        Initialize the dataset.

        Args:
            dataset: The input dataset (e.g., from Hugging Face datasets).
            tokenizer: Tokenizer for text processing (e.g., from transformers).
            image_processor: Processor for image preprocessing (e.g. form torchvision transforms).
            image_token_length: Number of tokens to represent each image.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_length = image_token_length

        self.prefix_length = self._compute_prefix_length()

    def __len__(self):
        return len(self.dataset)

    def _compute_prefix_length(self):
        """
        Compute the length of the tokenizer's prefix for a chat template.

        Uses a dummy string to determine the prefix length before the content in
        the chat template, which is needed for masking assistant responses.
        """
        dummy_content = "xzyvd"  # Arbitrary string to probe template structure
        templated_text = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": dummy_content}],
            tokenize=False,
            add_special_tokens=False,
        )
        content_start = templated_text.find(dummy_content)
        prefix_text = templated_text[:content_start]
        return len(self.tokenizer(prefix_text))

    def _build_messages(self, item, image_count=0):
        """
        Construct a list of chat messages from the dataset item.

        Args:
            item (dict): Dataset item containing 'texts' with user/assistant pairs.
            image_count (int): Number of images to prepend as tokens.

        Returns:
            list: List of message dictionaries with 'role' and 'content'.
        """
        messages = []
        for text in item["texts"]:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        # Prepend image tokens to the first message if images are present
        # Note: We only prepend the first image to the message
        if image_count > 0:
            image_tokens = self.tokenizer.image_token * self.image_token_length
            messages[0]["content"] = image_tokens + messages[0]["content"]

        return messages

    def _process_images(self, images):
        """
        Preprocess a list of images for model input.

        Args:
            images (list): List of PIL Image objects.

        Returns:
            list: List of preprocessed images.
        """
        processed_images = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("Expected PIL Image object")
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = self.image_processor(image)
            processed_images.append(processed_image)
        return processed_images

    def _prepare_inputs_and_loss_mask(self, messages):
        """
        Tokenize messages and create input IDs, loss mask, and attention mask.

        The loss mask is set to 1 for assistant response tokens (after prefix)
        to focus loss computation on assistant content.

        Args:
            messages (list): List of message dictionaries.

        Returns:
            tuple: (input_ids, loss_mask, attention_mask) as tensors.
        """
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        loss_mask = [0] * len(input_ids)  # Initialize loss mask with zeros

        # Track position in the tokenized sequence
        token_position = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            segment_length = len(segment_ids)

            if msg["role"] == "assistant":
                # Set loss mask to 1 for assistant tokens (after prefix)
                start = token_position + self.prefix_length
                end = token_position + segment_length
                loss_mask[start:end] = [1] * (end - start)

            token_position += segment_length

        return (
            torch.tensor(input_ids),
            torch.tensor(loss_mask, dtype=torch.bool),
            torch.tensor(attention_mask),
        )


class VQADataset(BaseDataset):
    """Dataset for Visual Question Answering, extending BaseDataset."""

    def __getitem__(self, idx):
        """
        Get a processed sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing images, input_ids, attention_mask, and labels.
        """
        item = self.dataset[idx]

        # Ensure images are a list
        images = item["images"]
        if not isinstance(images, list):
            images = [images]

        # Process images
        processed_images = self._process_images(images)

        # Build messages with image tokens
        messages = self._build_messages(item, len(processed_images))

        # Tokenize and create masks
        input_ids, loss_mask, attention_mask = self._prepare_inputs_and_loss_mask(
            messages
        )
        labels = self._compute_labels(input_ids, loss_mask)

        return {
            "image": processed_images[
                0
            ],  # Note: Taking only the first image in the list of images
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _compute_labels(self, input_ids, loss_mask):
        """
        Compute labels for causal language modeling, masked by loss_mask.

        Labels are shifted input_ids, with non-assistant tokens set to -100
        to ignore them in loss computation.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            loss_mask (torch.Tensor): Boolean mask for assistant tokens.

        Returns:
            torch.Tensor: Labels for training.
        """
        labels = input_ids.clone()
        labels[~loss_mask] = -100  # Ignore non-assistant tokens
        labels = labels.roll(-1)  # Shift for causal LM
        labels[-1] = -100  # No target for last token
        return labels
