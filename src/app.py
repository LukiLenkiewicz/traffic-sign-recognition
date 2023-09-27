import gradio as gr
import numpy as np
import torch
from torchvision import transforms

from model import CNN
from lightning_modules import CNNModule

def predict(input_img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
    ])

    module = CNNModule(CNN())

    model_trained = module.load_from_checkpoint("best-model.ckpt")
    model_trained.eval()

    input_img = transform(input_img)

    result = model_trained(input_img.unsqueeze(0).cuda())

    return str(torch.argmax(result).item())

demo = gr.Interface(predict, gr.Image(type="pil"), "text")
demo.launch()
