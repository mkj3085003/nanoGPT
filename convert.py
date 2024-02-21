import os
import torch
from model import GPTConfig, GPT
import onnx
from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
import keras
import tensorflow as tf
import tiktoken

batch_size = 64
block_size = 256 # context of up to 256 previous characters
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-char' # ignored if init_from is not 'resume'
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# compile = False # use PyTorch 2.0 to compile the model to be faster
# exec(open('configurator.py').read()) # overrides from command line or config file



# model
def load_model():
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    model.eval()
    print(model)
    return model
    




def pth_to_onnx(torch_model,output_path):
    '''
    1)声明：使用本函数之前，必须保证你手上已经有了.pth模型文件.
    2)功能：本函数功能四将pytorch训练得到的.pth文件转化为onnx文件。
    '''
    # torch_model_pt = torch.load(input_path)       # pytorch模型加载,此处加载的模型包含图和参数
    # torch_model= load_model()
    # # torch_model = selfmodel()  # 若只保存参数，selfmodel参考上述loadmodel进行改写
    # torch_model.eval()
    # print(torch_model)
    # enc = tiktoken.get_encoding("gpt2")
    # encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    # # start = "\n"
    # # start_ids = encode(start)
    # # x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])    
    # # print(x.size())
    x = torch.zeros(batch_size,block_size).long()       
    print(x.size())
    print(x)
    export_onnx_file = output_path         #输出.onnx文件的文件路径及文件名
    torch.onnx.export(torch_model,
                      x,
                      export_onnx_file,
                      opset_version=14,    #操作的版本，稳定操作集为9
                      do_constant_folding=True,          # 是否执行常量折叠优化
                      input_names=["input"],        # 输入名
                      output_names=["output"]      # 输出名
                      # dynamic_axes={"input": {0: "batch_size"},         # 批处理变量
                      #               "output": {0: "batch_size"}}
                      )
    onnx_model = onnx.load('model_all.onnx')    #加载.onnx文件
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))       #打印.onnx文件信息

def onnx_to_pb(output_path):
    '''
    将.onnx模型保存为.pb文件模型
    '''
    model = onnx.load(output_path) #加载.onnx模型文件
    tf_rep = prepare(model)
    tf_rep.export_graph('model_all.pb')    #保存最终的.pb文件

def onnx_to_h5(output_path ):
    '''
    将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
    '''
    onnx_model = onnx.load(output_path)
    k_model = onnx_to_keras(onnx_model, ['input'])
    keras.models.save_model(k_model, 'kerasModel.h5', overwrite=True, include_optimizer=True)    #第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = tf.keras.models.load_model('kerasModel.h5')
    model.summary()
    print(model)
    
if __name__=='__main__':
    model= load_model()
    model.eval()
    print(model)
    print(model.config)
    input_path = "nanoGPT/out-shakespeare-char/ckpt.pt"    #输入需要转换的.pth模型路径及文件名
    output_path = "model_all.onnx"  #转换为.onnx后文件的保存位置及文件名
    pth_to_onnx(model,output_path)  #执行pth转onnx函数，具体转换参数去该函数里面修改
    # onnx_pre(output_path)   #【可选项】若有需要，可以使用onnxruntime进行部署测试，看所转换模型是否可用，其中，output_path指加载进去的onnx格式模型所在路径及文件名
    onnx_to_pb(output_path)   #将onnx模型转换为pb模型
    onnx_to_h5(output_path)   #将onnx模型转换为h5模型

