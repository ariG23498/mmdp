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


def visualize_knapsack_packing(batch, seq_length, title, fname):
    """
    Visualizes how multiple samples are packed into constant-length sequences.
    """
    attention_masks = batch["attention_masks"]
    labels = batch["labels"]
    batch_size, max_len = attention_masks.shape

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(batch_size)

    for i in range(batch_size):
        # Find where sequences start and end using the labels (-100 is padding/separator)
        is_content = (labels[i] != -100).cpu().numpy()

        # Find changes from non-content to content
        starts = np.where(np.diff(np.concatenate(([False], is_content))) == 1)[0]
        # Find changes from content to non-content
        ends = np.where(np.diff(np.concatenate((is_content, [False]))) == -1)[0]

        colors = plt.cm.viridis(np.linspace(0, 1, len(starts)))

        for j, (start, end) in enumerate(zip(starts, ends)):
            length = end - start
            ax.barh(
                y_pos[i],
                length,
                left=start,
                height=0.6,
                align="center",
                color=colors[j],
                edgecolor="black",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Packed Seq {i + 1}" for i in range(batch_size)])
    ax.invert_yaxis()
    ax.set_xlabel("Token Position")
    ax.set_title(title)
    ax.set_xlim(0, max_len)
    ax.axvline(
        x=seq_length, color="r", linestyle="--", label=f"Target Length ({seq_length})"
    )

    # Add a dummy legend for colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=plt.cm.viridis(0.5), edgecolor="black", label="Individual Sample"
        )
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved knapsack packing visualization to {fname}")
