from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import typer

from data_utils import get_image_paths

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_images(self, images_path: Path, target_path: Path):
        images_dict = get_image_paths(images_path)
        self.create_directiories(images_dict, target_path)

        for sample in tqdm(images_dict):
            img = Image.open(sample["image"]).convert('RGB')
            img = self.transform(img)
            new_image_path = target_path / str(sample['label']) / sample['image'].parts[-1]
            img.save(new_image_path)

    def create_directiories(self, train_paths: dict, target_path: Path):
        label_paths = set()
        for path in train_paths:
            label_paths.add(path["label"])

        target_path.mkdir(exist_ok=True)
        for label_path in label_paths:
            new_path = target_path / str(label_path)
            new_path.mkdir(exist_ok=True)


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    typer.run(preprocessor.preprocess_images) 
