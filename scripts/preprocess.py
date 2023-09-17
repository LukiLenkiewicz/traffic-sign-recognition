from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from data_utils import get_train_paths

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_images(self, train_paths):
        for sample in tqdm(train_paths):
            img = Image.open(sample["image"]).convert('RGB')
            img = self.transform(img)
            new_image_path = Path(f"data/train_preprocessed/{sample['label']}/{sample['image'].parts[-1]}")
            print(new_image_path)
            img.save(new_image_path)

    def create_directiories(self):
        pass


if __name__ == "__main__":
    from pathlib import Path