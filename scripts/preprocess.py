from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from data_utils import get_train_paths

class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_images(self, train_paths: dict, target_path: Path):
        self.create_directiories(train_paths, target_path)

        for sample in tqdm(train_paths):
            img = Image.open(sample["image"]).convert('RGB')
            img = self.transform(img)
            new_image_path = target_path / sample['label'] / sample['image'].parts[-1]
            img.save(new_image_path)

    def create_directiories(self, train_paths: dict, target_path: Path):
        label_paths = set()
        for path in train_paths:
            label_paths.add(path["label"])

        target_path.mkdir(exist_ok=True)
        for label_path in label_paths:
            new_path = target_path / label_path
            new_path.mkdir(exist_ok=True)


if __name__ == "__main__":
    paths = get_train_paths(Path("/home/lukasz/traffic_sign_recognition/data/train"))

    preprocessor = ImagePreprocessor()
    preprocessor.preprocess_images(paths, target_path=Path("/home/lukasz/traffic_sign_recognition/data/preprocessed"))    
