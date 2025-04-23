# ✅ 修改后的 utils.py
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from config import CLIP_MODEL_NAME_OR_PATH
from PIL import Image
import torch

CLIP_MODEL_NAME_OR_PATH = "./clip-vit-base-patch32"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()


def load_image_model_processor():
    img_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    img_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME_OR_PATH)
    return img_model.to(device), img_processor

def encoder_text(text, model, tokenizer=None):
    return model.encode(text)

def encoder_image(image_path, model, processor):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    pooled_output = outputs.image_embeds
    if pooled_output is not None and pooled_output.shape[0] > 0:
        return list(pooled_output.cpu().detach().numpy()[0])
    return None