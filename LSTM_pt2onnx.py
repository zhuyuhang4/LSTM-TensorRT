import torch
import torch.nn as nn
test_data = torch.load('/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/test_data_back_0.pt')
Xtest, Ytest = test_data['Xtest'], test_data['Ytest']
# 定义模型（必须与训练时完全相同）
hidden_size = 200
class RNNModel(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):  
        super(RNNModel, self).__init__()  
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)  
        
    def forward(self, x, h0, c0):  
        out, _ = self.rnn(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

# 加载模型
model = RNNModel(input_size=Xtest.shape[2], hidden_size=200, output_size=1)  # 替换为你的input_size
model.load_state_dict(torch.load('/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/best_model_back_0.pt'))
model.eval()


batch_size = 1  # 用来导出 ONNX 的 dummy 数据 batch 大小
seq_len = Xtest.shape[1]
input_size = Xtest.shape[2]
hidden_size = 200  # 跟你模型里的一样

dummy_input = torch.randn(batch_size, seq_len, input_size)
h0_dummy = torch.zeros(1, batch_size, hidden_size)
c0_dummy = torch.zeros(1, batch_size, hidden_size)
# 转换为ONNX
onnx_path = "/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/model_back_0.onnx"

torch.onnx.export(
    model, 
    (dummy_input, h0_dummy, c0_dummy), 
    onnx_path,
    input_names=["input", "h0", "c0"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "h0": {1: "batch_size"},
        "c0": {1: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=11
)
print(f"Model saved to {onnx_path}")
