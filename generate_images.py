import multiprocessing as mp
import os
from functools import partial

import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_single_image(
    index, output_dir, image_size=(224, 224)
):
    """Generate a single random image and save it"""
    # Generate random pixel values (0-255)
    image_data = np.random.randint(
        0, 256, size=(*image_size, 3), dtype=np.uint8
    )

    # Convert to PIL Image
    img = Image.fromarray(image_data)

    # Save image with compression to save disk space
    filename = os.path.join(
        output_dir, f"random_image_{index:04d}.png"
    )
    img.save(filename, optimize=True)

    # Free memory
    del image_data
    del img


def main():
    # Create output directory
    output_dir = "random_images"
    os.makedirs(output_dir, exist_ok=True)

    # Number of CPU cores to use
    num_cores = max(
        1, mp.cpu_count() - 1
    )  # Leave one core free
    print(f"\nUsing {num_cores} CPU cores")

    # Create partial function with fixed arguments
    generate_func = partial(
        generate_single_image, output_dir=output_dir
    )

    # Use multiprocessing pool to generate images in parallel
    with mp.Pool(num_cores) as pool:
        # Create progress bar
        for _ in tqdm(
            pool.imap_unordered(generate_func, range(500)),
            total=500,
            desc="Generating images",
        ):
            pass


if __name__ == "__main__":
    main()
