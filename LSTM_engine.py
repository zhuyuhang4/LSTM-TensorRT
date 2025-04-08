import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import time
from scipy.stats import pearsonr

TRT_LOGGER = trt.Logger()

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def build_context_and_buffers(engine, input_shapes):
    context = engine.create_execution_context()

    # 设置动态输入 shape
    for idx, name in enumerate(engine):
        if engine.binding_is_input(name):
            context.set_binding_shape(idx, input_shapes[name])

    bindings = [None] * engine.num_bindings
    output_buffers = {}
    for idx, name in enumerate(engine):
        dtype = trt.nptype(engine.get_binding_dtype(name))
        shape = tuple(context.get_binding_shape(idx))  # ✅ 转为 Python tuple
        if engine.binding_is_input(name):
            bindings[idx] = None  # 稍后赋值
        else:
            out_tensor = torch.empty(shape, dtype=torch.float32, device="cuda")  
            bindings[idx] = int(out_tensor.data_ptr())
            output_buffers[name] = out_tensor
    stream = cuda.Stream()
    return context, bindings, output_buffers, stream


def infer_single_sample(context, bindings, output_buffers, stream, engine, inputs_dict):
    for idx, name in enumerate(engine):
        if engine.binding_is_input(name):
            tensor = inputs_dict[name].contiguous()
            bindings[idx] = int(tensor.data_ptr())

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    stream.synchronize()

    # 默认只输出第一个输出
    return output_buffers[list(output_buffers.keys())[0]].cpu().numpy()

# ==== 加载引擎 ====
engine_path = "/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/model_back_0.engine"
engine = load_engine(engine_path)

# ==== 读取测试数据 ====
test_data = torch.load('/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/test_data_back_0.pt')
Xtest, Ytest = test_data['Xtest'], test_data['Ytest']

# ==== 设置输入 shape ====
input_shapes = {
    "input": (1, 15, 141),
    "h0": (1, 1, 200),
    "c0": (1, 1, 200)
}

# ==== 初始化 context 和 buffers ====
context, bindings, output_buffers, stream = build_context_and_buffers(engine, input_shapes)

# ==== 准备初始状态 ====
h0_tensor = torch.zeros((1, 1, 200), dtype=torch.float32, device="cuda")
c0_tensor = torch.zeros((1, 1, 200), dtype=torch.float32, device="cuda")

# ==== 开始推理 ====
outputs = []
start = time.time()
for i in range(Xtest.shape[0]):
    x = Xtest[i].unsqueeze(0).cuda()  # (1, 15, 141)
    inputs_dict = {
        "input": x,
        "h0": h0_tensor,
        "c0": c0_tensor
    }
    output = infer_single_sample(context, bindings, output_buffers, stream, engine, inputs_dict)
    
    outputs.append(output)
end = time.time()

# ==== 评估 ====
y_pred = np.array(outputs).squeeze()
y_true = Ytest.cpu().numpy().flatten()
corr, p_value = pearsonr(y_true, y_pred)

print(f"TRT（GPU Tensor）推理耗时: {end - start:.2f} 秒")
print(f"Pearson Correlation: {corr:.4f}")
print(f"P-value: {p_value:.4g}")
