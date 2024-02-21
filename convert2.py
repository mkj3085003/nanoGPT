import tensorflow as tf
import tensorflowjs as tfjs

# 加载你的TensorFlow模型
model_path = "/root/nanoGPT/model_all.pb"  # 替换为你的模型文件路径
model = tf.saved_model.load(model_path)

# 转换模型
tfjs_model = tfjs.converters.convert(model, {
    'input_format': 'tf_saved_model',
    'output_format': 'tfjs_layers_model',
    'signature_keys': ['serving_default']
})

# 保存转换后的模型
tfjs_model.save('model.json')
