from PIL import Image
from matplotlib import pyplot as plt


def visualize_sample(sample, figsize=(12, 5), fname="sample.png"):
    """
    Visualize an image and its associated texts from a dataset sample.

    Args:
        sample (dict): Dataset sample containing 'images' and 'texts' keys
        figsize (tuple): Figure size as (width, height)
        fname (str): Filename to save the figure
    """
    # Create a figure with two subplots (image on left, text on right)
    _, (ax_img, ax_text) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1]}
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
