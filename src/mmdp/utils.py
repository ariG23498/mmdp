import numpy as np
from matplotlib import pyplot as plt


def visualize_sample(sample, fname="sample.png"):
    """
    Visualize an image and its associated texts from a dataset sample.

    Args:
        sample (dict): Dataset sample containing 'images' and 'texts' keys
        fname (str): Filename to save the figure
    """
    # Create a figure with two subplots (image on left, text on right)
    _, (ax_img, ax_text) = plt.subplots(
        1, 2, figsize=(10, 6), gridspec_kw={"width_ratios": [1, 1]}
    )

    # Plot the image (assuming single image in sample['images'])
    if sample["images"]:
        image = sample["images"][0]
        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title("Image")

    # Display texts (concatenate all text entries)
    if sample["texts"]:
        text_content = ""
        for text_item in sample["texts"]:
            text_content += f"\nUser: {text_item['user']}"
            text_content += f"\nAssistant: {text_item['assistant']}\n\n"

        # Display text in the right subplot
        ax_text.text(
            0.05,
            0.95,
            text_content.strip(),
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=10,
            wrap=True,
        )
        ax_text.axis("off")
        ax_text.set_title("Text Content")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved padding visualization to {fname}")


def visualize_padding(batch, max_len, title, fname):
    """
    Visualizes the padding in a batch of sequences.

    Args:
        batch (dict): A batch from a DataLoader.
        max_len (int): The maximum sequence length of the batch.
        title (str): The title for the plot.
        fname (str): The filename to save the plot.
    """
    input_ids = batch["input_ids"]
    attention_masks = batch["attention_masks"]

    batch_size = len(input_ids)
    seq_lengths = attention_masks.sum(dim=1).cpu().numpy()

    _, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(batch_size)

    # Plot the actual content length
    ax.barh(y_pos, seq_lengths, align="center", color="skyblue", label="Content")
    # Plot the padding
    ax.barh(
        y_pos,
        max_len - seq_lengths,
        left=seq_lengths,
        align="center",
        color="lightgrey",
        label="Padding",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Seq {i + 1}" for i in range(batch_size)])
    ax.invert_yaxis()
    ax.set_xlabel("Sequence Length")
    ax.set_title(title)
    ax.legend()

    plt.axvline(x=max_len, color="r", linestyle="--", label=f"Max Length ({max_len})")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved padding visualization to {fname}")
