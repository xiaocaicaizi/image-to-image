# insert_images_demo.py

import os
import torch
import numpy as np
from PIL import Image
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from transformers import CLIPProcessor, CLIPModel

# ----------- 配置 -----------
MILVUS_URI = "http://172.30.199.112:19530"
COLLECTION_NAME = "image_collection"
EMBEDDING_DIMENSION = 512
IMAGE_DIR = "images/"  # 替换为你要插入的图片文件夹
CLIP_MODEL_PATH = "./clip-vit-base-patch32"  # ✅ 使用你本地模型

# ----------- 加载本地模型 -----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)

def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs[0].cpu().numpy().astype(np.float32)

# ----------- Milvus 客户端类 -----------
class ImageMilvusClient:
    def __init__(self):
        self.client = MilvusClient(uri=MILVUS_URI)

        if not self.client.has_collection(COLLECTION_NAME):
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键字段
                FieldSchema(name="img_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION),
            ]

            # 定义 schema
            schema = CollectionSchema(fields, description="Image collection")

            # 创建 collection
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema
            )
            print(f"✅ 创建 collection: {COLLECTION_NAME}")

    def insert_images(self, image_folder):
        image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(image_folder)
            for file in files if file.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if not image_paths:
            print("⚠️ 未找到图片")
            return

        print(f"🔍 共发现 {len(image_paths)} 张图片，开始编码...")

        embeddings = [encode_image(p).tolist() for p in image_paths]

        data = [
            {"img_path": image_paths[i], "vector": embeddings[i]}
            for i in range(len(image_paths))
        ]

        res = self.client.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"✅ 成功插入 {len(data)} 条数据")
        print(f"🧾 插入返回: {res}")

# ----------- 执行入口 -----------
if __name__ == "__main__":
    client = ImageMilvusClient()
    client.insert_images(IMAGE_DIR)
