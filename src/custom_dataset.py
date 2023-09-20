from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image

class TrafficSignDatset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image"]
        image = Image.open(image_path).convert('RGB')
        image = ToTensor()(image)
        label = sample["label"]
        return image, label
