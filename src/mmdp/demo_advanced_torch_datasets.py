import itertools
import math
import threading
from queue import Queue
from typing import List

from torch.utils.data import IterableDataset, get_worker_info


class PackedDataset(IterableDataset):
    def __init__(
        self,
        start: int,
        end: int,
        max_length: int = 10,
        queue_size: int = 128,
        buffer_size: int = 512,
        strategy: str = "greedy",  # or "binpack"
    ):
        assert strategy in {"greedy", "binpack"}
        self.start = start
        self.end = end
        self.max_length = max_length
        self.queue_size = queue_size
        self.buffer_size = buffer_size
        self.strategy = strategy

    def _get_data_range(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single worker, return the entire dataset
            return self.start, self.end
        else:  # multiple workers, split the data load
            per_worker = int(
                math.ceil((self.end - self.start) / worker_info.num_workers)
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
            return iter_start, iter_end

    def _greedy_packing(self, iterator):
        pack, pack_sum = [], 0
        for item in iterator:
            if item > self.max_length:
                continue
            if pack_sum + item <= self.max_length:
                pack.append(item)
                pack_sum += item
            else:
                yield pack
                pack = [item]
                pack_sum = item
        if pack:
            yield pack

    def _bin_packing(self, buffer: List[int]):
        buffer = sorted(buffer, reverse=True)
        knapsacks = []

        for item in buffer:
            if item > self.max_length:
                continue
            placed = False
            for pack in knapsacks:
                if sum(pack) + item <= self.max_length:
                    pack.append(item)
                    placed = True
                    break
            if not placed:
                knapsacks.append([item])
        return knapsacks

    def _producer(self, data_iter, queue, stop_signal):
        if self.strategy == "greedy":
            for pack in self._greedy_packing(data_iter):
                queue.put(pack)
        elif self.strategy == "binpack":
            while True:
                buffer = list(itertools.islice(data_iter, self.buffer_size))
                if not buffer:
                    break
                knapsacks = self._bin_packing(buffer)
                for pack in knapsacks:
                    queue.put(pack)
        queue.put(stop_signal)

    def __iter__(self):
        iter_start, iter_end = self._get_data_range()
        data_iter = iter(range(iter_start, iter_end))

        queue = Queue(maxsize=self.queue_size)
        stop_signal = object()

        thread = threading.Thread(
            target=self._producer, args=(data_iter, queue, stop_signal)
        )
        thread.start()

        while True:
            pack = queue.get()
            if pack is stop_signal:
                break
            yield pack

        thread.join()
