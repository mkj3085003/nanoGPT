import tensorflow as tf
import tensorflowjs as tfjs

model_path = "/root/nanoGPT/model/model.json"  # 替换为你的模型路径
try:
    model = tfjs.loadGraphModel(model_path)
    print("模型加载成功。")
except Exception as e:
    print("模型加载失败：", e)