import os
from config import MILVUS_HOST, EMBEDDING_DIMENSION, MILVUS_COLLECTION_NAME
from utils import load_image_model_processor, encoder_image
from pymilvus import MilvusClient
import numpy as np

# 加载模型
image_model, image_processor = load_image_model_processor()

class MyMilvusClient:
    def __init__(self):
        self.client = MilvusClient(uri="http://172.30.199.112:19530")

    def store_image_data(self, image_data_dir):
        image_paths = []
        for root, dirs, files in os.walk(image_data_dir):
            for file in files:
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

        image_embeddings = [encoder_image(path, image_model, image_processor) for path in image_paths]

        # 使用自增的整数作为 id
        entities = [
            {"id": i, "vector": emb, "img_path": image_paths[i]}  # 使用自增整数作为 id
            for i, emb in enumerate(image_embeddings)
        ]

        insert_res = self.client.insert(collection_name=MILVUS_COLLECTION_NAME, data=entities)
        print(insert_res)

    def search_image(self, query, image_path, top_k=2):
        search_img_path = []
        query_vectors = []

        if image_path:
            image_query_vector = encoder_image(image_path, image_model, image_processor)
            query_vectors.append(image_query_vector)

        for query_vector in query_vectors:
            query_res = self.client.search(
                collection_name=MILVUS_COLLECTION_NAME,
                data=[query_vector],
                limit=top_k,
                output_fields=["img_path"],
                anns_field="vector",
            )
            for d in query_res[0]:
                search_img_path.append((d['entity']['img_path'], d['distance']))

        return [p for p, _ in search_img_path]

if __name__ == '__main__':
    milvus_client = MyMilvusClient()
    milvus_client.store_image_data("images/")  # 替换为你新增图片的目录