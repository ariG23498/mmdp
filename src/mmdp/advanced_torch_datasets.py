import itertools
import random
import threading
from queue import Queue
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import List

random.seed(42)  # Set the random seed to the meaning of life for good luck


class NaiveConstantLengthDataset(IterableDataset):
    """
    Wraps a `VQADataset` (or any map-style dataset that yields dicts with
    input_ids / labels / attention_mask / images) and emits *constant-length*
    training chunks built with a greedy knapsack algorithm.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The *tokenised* + *image-processed* base dataset (e.g. `VQADataset`).
    seq_length : int
        Target context length of each output chunk.
    num_sequences : int
        How many `seq_length` chunks to accumulate in one buffer before packing.
        Higher → better packing at the cost of more RAM / CPU time.
    max_sample_length : int
        Discard any raw sample whose tokenised length >= this value.
    max_images_per_example : int
        Discard samples with more than this many images.
    max_images_per_knapsack : int
        Upper bound on total images *inside one packed chunk*.
    delta : int
        Slack bins for the knapsack heuristic (larger delta → faster, slightly
        worse packing).
    infinite : bool
        If True, restart the base iterator when exhausted (useful for endless
        training).  Increments `self.epoch` each pass.
    """

    def __init__(
        self,
        dataset,
        *,
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
        self.epoch = 0  # only increases when infinite=True

        # crude length estimate (for __len__) → average text length + image tokens
        self._avg_tokens_per_sample = (
            self.dataset.image_token_length + 198
        )  # 198 = empirical for Cauldron

    # --------------------------------------------------------------------- #
    # PyTorch bookkeeping
    # --------------------------------------------------------------------- #
    def __len__(self):
        return int(len(self.dataset) * self._avg_tokens_per_sample / self.seq_length)

    # --------------------------------------------------------------------- #
    # Iterator
    # --------------------------------------------------------------------- #
    def __iter__(self):
        """Yield constant-length packed samples forever (or once if not infinite)."""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        def make_base_iter():
            indices = range(len(self.dataset))
            # shard indices across workers (handles num_workers > 1)
            if num_workers > 1:
                indices = itertools.islice(indices, worker_id, None, num_workers)
            for i in indices:
                yield self.dataset[i]

        iterator = make_base_iter()

        while True:
            # 1) Fill a buffer with raw samples until token budget reached
            buffer: List[dict] = []
            buf_len = 0
            while buf_len < self.max_length:
                try:
                    sample = next(iterator)
                except StopIteration:
                    if self.infinite:
                        iterator = make_base_iter()
                        self.epoch += 1
                        continue
                    else:
                        # done → flush what we have and exit outer loop
                        break

                if len(sample["input_ids"]) >= self.max_sample_length:
                    continue  # too long
                if len(sample["images"]) > self.max_images_per_example:
                    continue  # too many images

                # append a separator token so concatenated samples remain distinct
                sep_id = self.dataset.tokenizer.pad_token_id
                sample["input_ids"] = torch.cat(
                    [sample["input_ids"], torch.tensor([sep_id])]
                )
                sample["attention_mask"] = torch.cat(
                    [sample["attention_mask"], torch.tensor([0])]
                )
                sample["labels"] = torch.cat(
                    [sample["labels"], torch.tensor([-100])]
                )

                buffer.append(sample)
                buf_len += len(sample["input_ids"])

            if not buffer:
                break  # nothing left

            # 2) Greedy knapsack → list of groups (each fits seq_length + img cap)
            groups = self._balanced_greedy_knapsack(
                buffer,
                L=self.seq_length,
                delta=self.delta,
                max_images_per_knapsack=self.max_images_per_knapsack,
            )

            # 3) Pack each group into a constant-length chunk and yield
            for g in groups:
                ids, lbl, attn, imgs = self._pack_one_group(g, buffer, self.seq_length)
                yield {
                    "input_ids": ids,
                    "labels": lbl,
                    "attention_mask": attn,
                    "images": imgs,
                }

            if not self.infinite:
                break  # run only once when infinite=False

    # --------------------------------------------------------------------- #
    # Packing helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _balanced_greedy_knapsack(
        buffer: List[dict],
        L: int,
        *,
        delta: int = 0,
        max_images_per_knapsack: int | None = None,
    ) -> List[List[int]]:
        """Return list of index-groups; each group fits length & image budgets."""
        lengths = [len(x["input_ids"]) for x in buffer]
        img_counts = [len(x["images"]) for x in buffer]

        # sort by decreasing length, keep original index
        items = sorted(
            enumerate(zip(lengths, img_counts)),
            key=lambda x: x[1][0],
            reverse=True,
        )

        min_bins = (sum(lengths) + L - 1) // L + delta
        bin_load = [0] * min_bins
        bin_imgs = [0] * min_bins
        groups: List[List[int]] = [[] for _ in range(min_bins)]

        for idx, (tok_len, n_imgs) in items:
            target = None
            # try lightest bins first → more balanced fill
            for b in sorted(range(len(bin_load)), key=bin_load.__getitem__):
                fits_len = bin_load[b] + tok_len <= L
                fits_img = (
                    max_images_per_knapsack is None
                    or bin_imgs[b] + n_imgs <= max_images_per_knapsack
                )
                if fits_len and fits_img:
                    target = b
                    break
            if target is None:
                target = len(bin_load)
                bin_load.append(0)
                bin_imgs.append(0)
                groups.append([])
            groups[target].append(idx)
            bin_load[target] += tok_len
            bin_imgs[target] += n_imgs

        random.shuffle(groups)  # avoid length ordering bias
        return [g for g in groups if g]  # drop empties

    @staticmethod
    def _pack_one_group(group: List[int], buffer: List[dict], L: int):
        """Concatenate members of `group` and return constant-length tensors."""
        ids, lbl, attn, imgs = [], [], [], []
        for i in group:
            ids.extend(buffer[i]["input_ids"])
            lbl.extend(buffer[i]["labels"])
            attn.extend(buffer[i]["attention_mask"])
            imgs.extend(buffer[i]["images"])

        if len(ids) > L:
            raise ValueError(f"Packed length {len(ids)} exceeds limit {L}")

        # convert lists → tensors and pad on the *left* (in-place training style)
        pad_id = buffer[0]["input_ids"].new_full((L - len(ids),), self_pad := 0)
        pad_lbl = buffer[0]["labels"].new_full((L - len(lbl),), -100)
        pad_att = buffer[0]["attention_mask"].new_full((L - len(attn),), 0)

        ids = torch.cat([pad_id, torch.stack(ids)])
        lbl = torch.cat([pad_lbl, torch.stack(lbl)])
        attn = torch.cat([pad_att, torch.stack(attn)])

        return ids, lbl, attn, imgs


# ────────────────────────────────────────────────────────────────────────────────
# 2.  Producer–consumer version
# ────────────────────────────────────────────────────────────────────────────────
class ConstantLengthDataset(NaiveConstantLengthDataset):
    """
    Same greedy-knapsack packing as NaiveConstantLengthDataset, but carried out
    **in a background thread with a bounded Queue** so that CPU preprocessing,
    image loading and knapsack calculations run *while* the GPU trains.
    """

    def __init__(
        self,
        *args,
        queue_size: int = 2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.queue_size = queue_size
        self._sentinel = object()           # signals producer finished

    # ------------------------------------------------------------------ ITERATOR
    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        def make_base_iter():
            """Yield raw samples, sharded across DataLoader workers."""
            indices = range(len(self.dataset))
            if num_workers > 1:
                indices = itertools.islice(indices, worker_id, None, num_workers)
            for i in indices:
                yield self.dataset[i]

        q: Queue = Queue(maxsize=self.queue_size)
        prod = threading.Thread(
            target=self._producer, args=(make_base_iter, q), daemon=True
        )
        prod.start()

        while True:
            item = q.get()
            if item is self._sentinel:      # producer ran out of data
                break
            yield item

    # --------------------------------------------------------------- PRODUCER 👷‍♂️
    def _producer(self, make_iter, q: Queue):
        iterator = make_iter()
        more = True

        while more:
            # 1) Fill an in-RAM buffer with ≤ `max_length` tokens
            buf: List[dict] = []
            buf_len = 0
            while buf_len < self.max_length:
                try:
                    ex = next(iterator)
                except StopIteration:
                    if self.infinite:
                        iterator = make_iter()
                        self.epoch += 1
                        continue
                    more = False
                    break

                if len(ex["input_ids"]) >= self.max_sample_length:
                    continue
                if len(ex["images"]) > self.max_images_per_example:
                    continue

                # separator so concatenations stay distinct
                pad_id = self.dataset.tokenizer.pad_token_id
                ex["input_ids"] = torch.cat([ex["input_ids"], torch.tensor([pad_id])])
                ex["attention_mask"] = torch.cat([ex["attention_mask"], torch.tensor([0])])
                ex["labels"] = torch.cat([ex["labels"], torch.tensor([-100])])

                buf.append(ex)
                buf_len += len(ex["input_ids"])

            if not buf:
                break

            # 2) Greedy knapsack → constant-length chunks
            groups = self._balanced_greedy_knapsack(
                buf,
                L=self.seq_length,
                delta=self.delta,
                max_images_per_knapsack=self.max_images_per_knapsack,
            )

            for g in groups:
                ids, lbl, attn, imgs = self._pack_one_group(g, buf, self.seq_length)
                q.put(
                    {
                        "input_ids": ids,
                        "labels": lbl,
                        "attention_mask": attn,
                        "images": imgs,
                    }
                )

        q.put(self._sentinel)               # unblock consumer when done