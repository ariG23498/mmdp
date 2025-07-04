import random
import threading
from queue import Queue
from typing import Iterator, List

import torch
from torch.utils.data import IterableDataset

random.seed(42)  # Set the random seed for reproducibility


class NaiveConstantLengthDataset(IterableDataset):
    """
    Wraps a map-style dataset to create an iterable dataset of constant-length
    training chunks. This is achieved by greedily packing variable-length
    samples into fixed-size chunks using a knapsack-like algorithm.

    This "naive" version performs all data fetching, processing, and packing
    within the main dataloader worker's iteration loop.

    Args:
        dataset (torch.utils.data.Dataset): The base dataset, which should be
            tokenized and have image processing applied (e.g., `VQADataset`).
        seq_length (int): The target sequence length for each packed chunk.
        num_sequences (int): The number of sequences to buffer before packing.
            A larger buffer can lead to better packing efficiency at the cost of
            increased memory usage.
        max_sample_length (int): Raw samples with tokenized lengths greater
            than this value will be discarded.
        max_images_per_example (int): Samples with more images than this value
            will be discarded.
        max_images_per_knapsack (int): The maximum total number of images
            allowed within a single packed chunk.
        delta (int): An adjustment factor for the number of bins in the
            knapsack algorithm. A larger delta can speed up packing at the cost
            of slightly lower efficiency.
        infinite (bool): If True, the dataset will loop indefinitely. The
            `self.epoch` attribute is incremented on each pass.
    """

    def __init__(
        self,
        dataset,
        seq_length: int = 1024,
        num_sequences: int = 1024,
        max_sample_length: int = 1024,
        max_images_per_example: int = 4,
        max_images_per_knapsack: int = 18,
        delta: int = 5,
        infinite: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.max_length = seq_length * num_sequences
        self.max_sample_length = max_sample_length
        self.max_images_per_example = max_images_per_example
        self.max_images_per_knapsack = max_images_per_knapsack
        self.delta = delta
        self.infinite = infinite
        self.epoch = 0

    def __iter__(self):
        """Yields constant-length packed samples."""
        iterator = iter(self.dataset)

        while True:
            buffer, buffer_len = self._fill_buffer(iterator)

            if not buffer:
                if not self.infinite:
                    break
                iterator = iter(self.dataset)
                self.epoch += 1
                continue

            groups = self._pack_buffer(buffer)

            for group in groups:
                yield self._format_group(group, buffer)

            if not self.infinite and not buffer:
                break

    def _fill_buffer(self, iterator):
        """Fills a buffer with samples from the dataset iterator."""
        buffer: List[dict] = []
        buffer_len = 0
        while buffer_len < self.max_length:
            try:
                sample = next(iterator)
            except StopIteration:
                break

            if (
                len(sample["input_ids"]) >= self.max_sample_length
                or len(sample["images"]) > self.max_images_per_example
            ):
                continue

            self._add_separator_token(sample)
            buffer.append(sample)
            buffer_len += len(sample["input_ids"])

        return buffer, buffer_len

    def _add_separator_token(self, sample):
        """Adds a separator token to the end of a sample's sequences."""
        sep_id = self.dataset.tokenizer.pad_token_id
        sample["input_ids"] = torch.cat([sample["input_ids"], torch.tensor([sep_id])])
        sample["attention_mask"] = torch.cat(
            [sample["attention_mask"], torch.tensor([0])]
        )
        sample["labels"] = torch.cat([sample["labels"], torch.tensor([-100])])

    def _pack_buffer(self, buffer):
        """Packs the buffer into groups using a greedy knapsack algorithm."""
        return self._balanced_greedy_knapsack(
            buffer,
            L=self.seq_length,
            delta=self.delta,
            max_images_per_knapsack=self.max_images_per_knapsack,
        )

    def _format_group(self, group, buffer):
        """Formats a group of samples into a single packed dictionary."""
        ids, lbl, attn, imgs = self._pack_one_group(group, buffer, self.seq_length)
        return {
            "input_ids": ids,
            "labels": lbl,
            "attention_mask": attn,
            "images": imgs,
        }

    @staticmethod
    def _balanced_greedy_knapsack(
        buffer: List[dict],
        L: int,
        *,
        delta: int = 0,
        max_images_per_knapsack: int | None = None,
    ) -> List[List[int]]:
        """
        Packs items into bins of capacity L using a greedy approach.

        Returns a list of groups, where each group is a list of indices into the buffer.
        """
        lengths = [len(x["input_ids"]) for x in buffer]
        img_counts = [len(x["images"]) for x in buffer]
        items = sorted(
            enumerate(zip(lengths, img_counts)), key=lambda x: x[1][0], reverse=True
        )

        min_bins = (sum(lengths) + L - 1) // L + delta
        bins = [{"load": 0, "images": 0, "indices": []} for _ in range(min_bins)]

        for idx, (tok_len, n_imgs) in items:
            target_bin = None
            for b in sorted(range(len(bins)), key=lambda i: bins[i]["load"]):
                if bins[b]["load"] + tok_len <= L and (
                    max_images_per_knapsack is None
                    or bins[b]["images"] + n_imgs <= max_images_per_knapsack
                ):
                    target_bin = b
                    break

            if target_bin is None:
                bins.append({"load": 0, "images": 0, "indices": []})
                target_bin = len(bins) - 1

            bins[target_bin]["indices"].append(idx)
            bins[target_bin]["load"] += tok_len
            bins[target_bin]["images"] += n_imgs

        groups = [b["indices"] for b in bins if b["indices"]]
        random.shuffle(groups)
        return groups

    @staticmethod
    def _pack_one_group(group: List[int], buffer: List[dict], L: int):
        """Concatenates samples in a group and pads to a constant length."""
        ids, lbl, attn, imgs = [], [], [], []
        for i in group:
            ids.extend(buffer[i]["input_ids"])
            lbl.extend(buffer[i]["labels"])
            attn.extend(buffer[i]["attention_mask"])
            imgs.extend(buffer[i]["images"])

        if len(ids) > L:
            raise ValueError(f"Packed length {len(ids)} exceeds limit {L}")

        pad_id = buffer[0]["input_ids"].new_full((L - len(ids),), 0)
        pad_lbl = buffer[0]["labels"].new_full((L - len(lbl),), -100)
        pad_att = buffer[0]["attention_mask"].new_full((L - len(attn),), 0)

        ids_tensor = torch.cat([pad_id, torch.tensor(ids)])
        lbl_tensor = torch.cat([pad_lbl, torch.tensor(lbl)])
        attn_tensor = torch.cat([pad_att, torch.tensor(attn)])

        return ids_tensor, lbl_tensor, attn_tensor, imgs


class ConstantLengthDataset(NaiveConstantLengthDataset):
    """
    An optimized version of `NaiveConstantLengthDataset` that uses a background
    thread and a queue to pre-fetch and process data.

    This allows the data preparation (I/O, tokenization, packing) to run
    concurrently with model training on the GPU, which can significantly
    improve throughput by reducing data loading bottlenecks.

    Args:
        queue_size (int): The maximum number of packed samples to store in the
            producer-consumer queue.
        (Other args are inherited from `NaiveConstantLengthDataset`)
    """

    def __init__(
        self,
        *args,
        queue_size: int = 2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.queue_size = queue_size
        self._sentinel = object()  # Signal for the producer to stop

    def __iter__(self) -> Iterator[dict]:
        """
        Starts a producer thread and yields packed samples from the queue.
        """
        q: Queue = Queue(maxsize=self.queue_size)
        producer_thread = threading.Thread(
            target=self._producer, args=(q,), daemon=True
        )
        producer_thread.start()

        while True:
            item = q.get()
            if item is self._sentinel:
                break
            yield item

    def _producer(self, q: Queue):
        """
        The producer function that runs in a separate thread.
        It fetches, processes, and packs data, then puts the packed samples
        into the queue for the consumer.
        """
        iterator = iter(self.dataset)

        while True:
            buffer, buffer_len = self._fill_buffer(iterator)

            if not buffer:
                if not self.infinite:
                    break
                iterator = iter(self.dataset)
                self.epoch += 1
                continue

            groups = self._pack_buffer(buffer)

            for group in groups:
                q.put(self._format_group(group, buffer))

            if not self.infinite and not buffer:
                break

        q.put(self._sentinel)
