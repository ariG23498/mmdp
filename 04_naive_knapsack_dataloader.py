from torch.utils.data import DataLoader

from mmdp.demo_advanced_torch_datasets import PackedDataset

if __name__ == "__main__":
    for strategy in ["greedy", "binpack"]:
        print(f"\n=== Strategy: {strategy.upper()} ===\n")
        dataset = PackedDataset(start=1, end=25, max_length=100, strategy=strategy)
        dataloader = DataLoader(dataset)

        for batch in dataloader:
            print(batch)
