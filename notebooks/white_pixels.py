import os
import torch
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

def count_white_pixels(image_tensor):
    return torch.sum(image_tensor == 1.0).item()

def process_images(root_dir, transform=None):
    total_white_pixels = 0
    total_pixels = 0

    for subdir, _, files in os.walk(root_dir):
        for file in tqdm(files, desc=f"Processing {subdir}"):
            file_path = os.path.join(subdir, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = Image.open(file_path).convert('L')  # Ensure grayscale
                    if transform:
                        image = transform(image)
                    else:
                        image = T.ToTensor()(image)


                    total_white_pixels += count_white_pixels(image)
                    total_pixels += image.numel()
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    if total_pixels == 0:
        return 0
    return total_white_pixels / total_pixels

if __name__ == "__main__":

    root_dir =  r"C:\Users\Everyday\Downloads\113094\ZooCamNet\imgs"

    custom_transforms = T.Compose([
        T.ToTensor(),
        T.Resize(223,max_size= 224, antialias=True),
        T.Pad(112, fill=255, padding_mode='constant'),
        T.CenterCrop((224, 224)),
    ])

    overall_mean = process_images(root_dir, transform=custom_transforms)

    print(f"Overall mean of white pixels across all images: {overall_mean:.6f}")

