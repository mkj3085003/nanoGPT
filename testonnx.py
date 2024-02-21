import onnxruntime as ort
import torch
import os
import pickle
from torch.nn import functional as F
import numpy as np
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
batch_size = 64
block_size = 256 # context of up to 256 previous characters

def generate(idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        # logits, _ = self(idx_cond)
        input_data = {'input': idx.cpu().numpy()}  # 将输入数据转换为 NumPy 数组
        outputs = session.run(None, input_data)
        logits = torch.tensor(outputs)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
    

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 加载模型
model_path = "/root/nanoGPT/model_all.onnx"
session = ort.InferenceSession(model_path)

# 加载元数据
meta_path = "/root/nanoGPT/data/shakespeare_char/meta.pkl"
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta.get('stoi', {}), meta.get('itos', {})

    # 定义编码和解码函数
    def encode(s):
        return [stoi.get(c, 0) for c in s]  # 使用 0 作为未知字符的索引

    def decode(l):
        return ''.join([itos.get(i, '') for i in l])  # 未知字符为空字符串

    # 准备输入数据
    start = "\n"
    start_ids = encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    input_data = {'input': x.cpu().numpy()}  # 将输入数据转换为 NumPy 数组
    outputs = session.run(None, input_data)
    print(outputs) 
    outputs_array = np.array(outputs)
    print(outputs_array.shape)
#     # 运行模型并打印结果
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             y = generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             print(decode(y[0].tolist()))
#             print('---------------')
