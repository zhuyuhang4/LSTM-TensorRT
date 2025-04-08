import onnxruntime as ort
import numpy as np
import torch
from scipy.stats import pearsonr

# 1. 加载ONNX模型
onnx_path = "/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/model_back_0.onnx"
ort_session = ort.InferenceSession(onnx_path)

# 2. 加载测试数据
test_data = torch.load('/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/test_data_back_0.pt')
Xtest, Ytest = test_data['Xtest'], test_data['Ytest']

# 3. 准备输入
Xtest_np = Xtest.cpu().numpy().astype(np.float32)
batch_size = Xtest_np.shape[0]
hidden_size = 200  # 必须跟训练时一致

# 构造 h0, c0（注意 shape 必须是 [num_layers, batch, hidden]，这里默认是 1 层）
h0_np = np.zeros((1, batch_size, hidden_size), dtype=np.float32)
c0_np = np.zeros((1, batch_size, hidden_size), dtype=np.float32)

# 4. 推理
onnx_inputs = {
    "input": Xtest_np,
    "h0": h0_np,
    "c0": c0_np
}
onnx_outputs = ort_session.run(None, onnx_inputs)
predictions = onnx_outputs[0]

# 5. 相关性分析
y_true = Ytest.cpu().numpy().flatten()
y_pred = predictions.flatten()

corr, p_value = pearsonr(y_true, y_pred)

print("ONNX预测结果形状:", predictions.shape)
print(f"Pearson Correlation: {corr:.4f}")
print(f"P-value: {p_value:.4g}")
