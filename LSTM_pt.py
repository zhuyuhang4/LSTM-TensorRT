import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
hidden_size=200
# 1. 定义模型（必须与训练时相同）
class RNNModel(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):  
        super(RNNModel, self).__init__()  
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)  
        
    def forward(self, x):  
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device) 
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

# 2. 加载测试数据
test_data = torch.load('/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/test_data_back_0.pt')
Xtest, Ytest = test_data['Xtest'], test_data['Ytest']

# 3. 加载模型
model = RNNModel(input_size=Xtest.shape[2], hidden_size=200, output_size=1)
model.load_state_dict(torch.load('/BMI/Project-NC-2023-A-02/1 - data/Rats/R902/20240111/Result/best_model_back_0.pt'))
model.eval()

# 4. 使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
Xtest, Ytest = Xtest.to(device), Ytest.to(device)
start_time = time.time()
# 5. 预测
with torch.no_grad():
    for i in range(Xtest.shape[0]):
        predictions = model(Xtest[i].unsqueeze(0))
        print("Inference result:", predictions)
end_time = time.time()

# --- 计算时间 ---
total_time = end_time - start_time
print(total_time)
