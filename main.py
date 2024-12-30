import requests
from io import BytesIO
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset


DATASET_NAME = "ceyda/fashion-products-small"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_function(examples):
    images = []
    for image in examples['image']:

        if not isinstance(image, Image.Image):

            if image.startswith("http"):
                response = requests.get(image, stream=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image)
        images.append(image)

    examples['pixel_values'] = clip_processor(images=images, return_tensors="pt").pixel_values
    return examples

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(preprocess_function, batched=True, batch_size=BATCH_SIZE)



class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, image_inputs):
      image_features = self.clip_model.get_image_features(**image_inputs)
      return image_features

model = CustomCLIP(clip_model).to("cuda" if torch.cuda.is_available() else "cpu") # Move model to GPU if available
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CosineEmbeddingLoss()


def get_description_and_score(image_path):

    if image_path.startswith("http"):
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu") # Move input to GPU

    with torch.no_grad():
      image_features = model(inputs)


    blip_inputs = blip_processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    out = blip_model.generate(**blip_inputs)
    text_description = blip_processor.decode(out[0], skip_special_tokens=True)


    text_inputs = clip_processor(text=[text_description], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    text_features = clip_model.get_text_features(**text_inputs)
    similarity_score = torch.cosine_similarity(image_features, text_features).item()

    return text_description, similarity_score



image_path="/content/drive/MyDrive/Colab Notebooks/beautiful-bouquet-with-different-flowers-roses-and-chrysanthemums-colorful-bouquet-of-different-fresh-flowers-close-up-2AKYDEM.jpg"

description, score = get_description_and_score(image_path)
print("Generated Caption:-", description)
print("Similarity Score:-", score)




# Author: Shrinkhla
# Copyright (C) [2024]
# All rights reserved.

