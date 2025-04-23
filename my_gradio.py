import gradio as gr
from my_milvus_client import MyMilvusClient

title = "Multimodal Search (Text ↔ Image & Text ↔ Text)"
description = "支持以图搜图、以文搜图、以及以文搜文的 Milvus + CLIP 系统"

milvus_client = MyMilvusClient()

# 图像搜索接口
def search_image(query, image_path, num_results):
    results = milvus_client.search_image(query, image_path, top_k=num_results)
    return results if results else ["No results found."]



# 文本/图像搜索界面
textbox = gr.Textbox(label="Type your query here")
image_upload = gr.Image(label="Upload an image to search", type="filepath")
slider = gr.Slider(label="Number of results", minimum=3, maximum=12, step=1)

image_interface = gr.Interface(
    fn=search_image,
    inputs=[textbox, image_upload, slider],
    outputs=gr.Gallery(label="Search Results"),
    title="Text-Image & Image-Image Search",
    description="输入文本或上传图片，搜索相关图像。"
)



# 合并两个界面
gr.TabbedInterface(
    [image_interface],
    tab_names=["以图/文搜图"]
).launch()
