import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
from pathlib import Path
import random
import csv

# 设置随机种子以确保代码每次运行的一致性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = False # 修改为False以允许非确定性操作，可能会提高训练速度
    torch.backends.cudnn.benchmark = True # 启用以允许cuDNN自动优化

# 初始化网络权重，使用Xavier初始化权重和零偏差，以保证权重在合理范围内，促进收敛
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# 构建多层全连接网络，末层不使用Tanh激活
class CustomNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super(CustomNetwork, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers[:-1])  # remove the last Tanh
        self.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

class EmbeddedCustomNetwork(nn.Module):
    def __init__(self, input_size, state_size):
        super(EmbeddedCustomNetwork, self).__init__()
        # 将3个可能的模式变量映射到5维空间
        self.mode_embedding = nn.Embedding(3, 5)  

        # 设定真正模型层的尺寸，考虑到mode的嵌入向量与原始input拼接
        embedded_input_size = input_size + 5  # 输入加上嵌入向量的维度

        # 设置层大小，模拟"[input_size + state_size] + [64] * 6 + [state_size]"结构
        layer_sizes = [embedded_input_size + state_size] + [64] * 6 + [state_size]
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # 最后一个线性层之前添加Tanh激活函数
                layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)  # 初始化权重

    def forward(self, x, mode):
        # 将模式编码为嵌入向量
        mode_embed = self.mode_embedding(mode)
        # 将模式嵌入向量与连续输入拼接
        x = torch.cat([x, mode_embed], dim=-1)
        # 通过模型层传递数据
        return self.layers(x)

# 加载转换后的MATLAB文件并返回PyTorch张量格式的输入、状态和输出数据，以及缩放因子
def load_data(path):
    # 加载转换后的MATLAB文件
    MatlabData = scipy.io.loadmat(path)
    # 提取数据
    U_array = MatlabData['U_array'][0]
    Y_array = MatlabData['Y_array'][0]

    # 根据指定的列索引选择列
    U_array = [u[:, [0, 5, 6, 7]] for u in U_array]
    Y_array = [y[:, [3, 4, 13, 14]] for y in Y_array]

    scalingFactors = MatlabData['scalingFactors']

    # 转换为 PyTorch 张量
    U_torch = [torch.tensor(U_array[i], dtype=torch.float32) for i in range(len(U_array))]
    Y_torch = [torch.tensor(Y_array[i], dtype=torch.float32) for i in range(len(Y_array))]
    return U_torch, Y_torch, scalingFactors

# 自定义数据集类
class ExperimentDataset(Dataset):
    def __init__(self, U, Y, sampling_interval):
        self.U = [u[::sampling_interval, :] for u in U]     # 欠采样的输入
        self.X = [y[::sampling_interval, 0:2] for y in Y]   # 欠采样的状态
        self.Y = [y[::sampling_interval, 2:] for y in Y]    # 欠采样的输出

    def __len__(self):
        return len(self.U)

    def __getitem__(self, idx):
        return self.U[idx], self.X[idx], self.Y[idx]

# 绘图逻辑的函数
def plot_predictions(epoch, time, states_pred, X, output_pred, Y, state_size, output_size, folder):
    # 确保文件夹存在
    folder.mkdir(parents=True, exist_ok=True)

    # Plotting X_pred (state predictions) against X (true states)
    plt.figure(figsize=(12, 6))
    for j in range(state_size):
        plt.subplot(state_size, 1, j+1)
        plt.plot(time, states_pred[:, j].detach().cpu().numpy(), label=f'x{j+1}_pred')
        plt.plot(time, X[:, j].detach().cpu().numpy(), label=f'x{j+1}', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel(f'X{j+1}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(folder / f'state_epoch_{epoch}.png')
    plt.close()

    # Plotting Y_pred (output predictions) against Y (true outputs)
    plt.figure(figsize=(12, 6))
    for k in range(output_size):
        plt.subplot(output_size, 1, k+1)
        plt.plot(time, output_pred[:, k].detach().cpu().numpy(), label=f'y{k+1}_pred')
        plt.plot(time, Y[:, k].detach().cpu().numpy(), label=f'y{k+1}', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel(f'Y{k+1}')
        plt.legend()

    plt.tight_layout()
    plt.savefig(folder / f'output_epoch_{epoch}.png')
    plt.close()

# 绘制损失曲线的函数
def plot_loss(epoch, state_losses, output_losses, folder, loss_plot_interval):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(state_losses) + 1)
    plt.plot(epochs, state_losses, label='State Network Loss')
    plt.plot(epochs, output_losses, label='Output Network Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss from Epoch {epoch - (loss_plot_interval - 1)} to {epoch}')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder / f'loss_epoch_{epoch}.png')
    plt.close()

# Merge overlapping or consecutive segments into larger segments.
def merge_segments(segments):
    if not segments:
        return segments
    
    # Sort segments by their start point
    segments.sort(key=lambda x: (x[0], x[1]))
    
    merged_segments = [segments[0]]
    for current in segments[1:]:
        last = merged_segments[-1]
        # Check if current segment overlaps or is consecutive with the last segment
        if current[0] == last[0] and (current[1] <= last[2] or current[1] - last[2] == 1):
            # Merge the two segments
            merged_segments[-1] = (last[0], min(last[1], current[1]), max(last[2], current[2]))
        else:
            merged_segments.append(current)

    return merged_segments

def find_segments(tensor, up_thd_scaled, low_thd_scaled, max_length=100):
    batch_size, time_step_size, _ = tensor.size()
    segments = []

    for batch in range(batch_size):
        state = tensor[batch, :, 0]  # Consider only the first state

        # Update the logic for defining zero and non-zero based on your new requirement
        is_zero = (state <= up_thd_scaled) & (state >= low_thd_scaled)
        not_zero = (state > up_thd_scaled) | (state < low_thd_scaled)

        # Find transitions
        zero_to_non_zero = torch.where(is_zero[:-1] & not_zero[1:])[0] + 1
        non_zero_to_zero = torch.where(not_zero[:-1] & is_zero[1:])[0] + 1

        for idx in zero_to_non_zero:
            startx = idx - 1
            while startx > 0 and is_zero[startx] and idx - startx < max_length:
                startx -= 1
            if not is_zero[startx] or idx - startx >= max_length:
                startx += 1  # Adjust to keep non-zero or respect max_length
            segments.append((batch, startx, idx))

        for idx in non_zero_to_zero:
            endx = idx + 0  # VERY IMPORTANT!!!
            while endx < time_step_size and is_zero[endx] and endx - idx < max_length:
                endx += 1
            if endx >= time_step_size or not is_zero[endx] or endx - idx >= max_length:
                endx -= 1
            segments.append((batch, idx, endx + 1))
        
        # Additional logic to capture transitions within max_length
        for start_idx in zero_to_non_zero:
            for end_idx in non_zero_to_zero:
                if 0 < end_idx - start_idx < max_length:
                    segments.append((batch, start_idx, end_idx + 1))
                    break  # Assuming you want to match each start_idx with only the first qualifying end_idx

    # Remove duplicates
    unique_segments = list(set(segments))
    
    # Merge overlapping or consecutive segments
    merged_segments = merge_segments(unique_segments)

    return merged_segments

# Zero out the segments in the tensor for the first state.
def apply_segments(tensor, segments):
    for batch, start, end in segments:
        tensor[batch, start:end, 0] = 0
    return tensor

# 逐个时间步长更新预测状态
def train_one_epoch(state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size, device, scalingFactors, sampling_interval, segments):
    state_loss_epoch = 0  # Record the state network loss for the epoch
    output_loss_epoch = 0  # Record the output network loss for the epoch
        
    for U_batch, X_batch, Y_batch in dataloader:
        U_batch, X_batch, Y_batch = U_batch.to(device), X_batch.to(device), Y_batch.to(device)
        
        state_optimizer.zero_grad()
        output_optimizer.zero_grad()

        # states_pred_batch 的初始值
        states_pred_batch = [X_batch[:, 0, :state_size]]

        for t in range(1, U_batch.shape[1]):
            state_input_batch = torch.cat((states_pred_batch[-1], U_batch[:, t-1, :]), dim=-1)
            
            dxdt_batch = state_net(state_input_batch)
            
            next_states_unclamped = states_pred_batch[-1] + dxdt_batch * dt * sampling_interval

            # 使用新的张量继续后续操作
            states_pred_batch.append(next_states_unclamped)

        states_pred_batch = torch.stack(states_pred_batch, dim=1)

        # Apply segments to both states_pred_batch and X_batch
        predictions = apply_segments(states_pred_batch.clone(), segments)
        ground_truth = apply_segments(X_batch[:, :, :state_size].clone(), segments)

        loss_state = criterion(predictions, ground_truth)

        # Gradient and optimization steps
        loss_state.backward(retain_graph=True)

        state_loss_epoch += loss_state.item() # 累积损失

        output_pred_batch = []
        for t in range(U_batch.shape[1]):
            output_input_batch = torch.cat((states_pred_batch[:, t, :], U_batch[:, t, 1:]), dim=-1)
            y_pred_batch = output_net(output_input_batch)
            output_pred_batch.append(y_pred_batch)

        output_pred_batch = torch.stack(output_pred_batch, dim=1)
        loss_output = criterion(output_pred_batch, Y_batch)
        loss_output.backward()
        
        state_optimizer.step()
        output_optimizer.step()
        output_loss_epoch += loss_output.item()

    return X_batch, Y_batch, states_pred_batch, output_pred_batch, state_loss_epoch, output_loss_epoch

# 运行指定的Epoch数量进行训练，每隔特定Epoch保存模型，绘制损失和输出预测
def training_loop(n_epochs, state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size, output_size, data_folder, device, scalingFactors, sampling_interval, segments):
    sample_plot_interval = 400
    loss_plot_interval = 2000
    PTH_save_interval = 4000

    # 定义用于存储损失的列表
    state_losses = []
    output_losses = []

    # 确保数据保存文件夹存在
    data_folder_path = Path(data_folder)
    data_folder_path.mkdir(parents=True, exist_ok=True)

    # CSV文件路径
    csv_file_path = data_folder_path / 'loss_and_duration.csv'

    # 使用'w'模式打开CSV文件以创建或覆盖现有文件
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV文件头
        writer.writerow(["Epoch", "State Loss", "Output Loss", "Duration(s)"])

        for epoch in range(n_epochs):
            # 开始计时
            start_time = time.time()
            X_batch, Y_batch, states_pred_batch, output_pred_batch, state_loss_epoch, output_loss_epoch = \
                train_one_epoch(state_net, output_net, state_optimizer, output_optimizer, criterion, dataloader, dt, state_size, device, scalingFactors, sampling_interval, segments)
            
            if (epoch + 1) % sample_plot_interval == 0 or epoch == n_epochs - 1 or epoch == 0:
                time_points = torch.arange(0, X_batch.shape[1]*dt*sampling_interval, dt*sampling_interval).cpu().numpy()
                plot_predictions(epoch + 1, time_points, states_pred_batch[5], X_batch[5].cpu(), output_pred_batch[5], Y_batch[5].cpu(), state_size, output_size, Path(data_folder))
            
            if (epoch + 1) % PTH_save_interval == 0 or epoch == n_epochs - 1:
                torch.save(state_net.state_dict(), Path(data_folder) / f'state_net_{epoch + 1}.pth')
                torch.save(output_net.state_dict(), Path(data_folder) / f'output_net_{epoch + 1}.pth')

            if (epoch + 1) % loss_plot_interval == 0 or epoch == n_epochs - 1:
                plot_loss(epoch + 1, state_losses, output_losses, Path(data_folder), loss_plot_interval)
                # 损失列表更新
                state_losses = []
                output_losses = []

            # Print average loss for the epoch and time taken
            end_time = time.time()
            epoch_duration = end_time - start_time
            # 写入当前epoch的数据到CSV文件
            writer.writerow([epoch+1, state_loss_epoch, output_loss_epoch, epoch_duration])
            print(f"Epoch {epoch+1}/{n_epochs}, State loss: {state_loss_epoch:.6f}, Output loss: {output_loss_epoch:.6f}, Duration: {epoch_duration:.2f}s")

            # 更新当前间隔内的损失列表
            state_losses.append(state_loss_epoch)
            output_losses.append(output_loss_epoch)

# 主程序入口点
if __name__ == "__main__":
    set_seed(42)
    # 定义网络结构并创建StateNetwork的实例
    input_size, state_size, output_size = 4, 2, 2
    state_net = CustomNetwork([input_size + state_size] + [64] * 6 + [state_size])
    output_net = CustomNetwork([input_size + state_size - 1] + [64] * 3 + [output_size])

    # 创建数据集实例(欠采样)
    U_torch, Y_torch, scalingFactors = load_data('./data/MatlabData.mat')
    sampling_interval = 1
    dataset = ExperimentDataset(U_torch, Y_torch, sampling_interval)
    # 创建数据加载器，batch_size等于U_torch中元素的数量
    dataloader = DataLoader(dataset, batch_size=len(U_torch), shuffle=False)

    x1_scale = scalingFactors[0][0]['LongitudinalVelocity_kph']['scale'][0, 0][0, 0]
    x1_offset = scalingFactors[0][0]['LongitudinalVelocity_kph']['offset'][0, 0][0, 0]
    up_thd_scaled = 0.5 * x1_scale + x1_offset # 上限
    low_thd_scaled = -0.5 * x1_scale + x1_offset # 下限

    # 定义优化器和损失函数
    state_optimizer = optim.Adam(state_net.parameters(), lr=5e-5, betas=(0.9, 0.99))
    output_optimizer = optim.Adam(output_net.parameters(), lr=5e-5, betas=(0.9, 0.99))
    criterion = nn.L1Loss() # 平均绝对误差

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_net.to(device)
    output_net.to(device)

    for U_batch, X_batch, Y_batch in dataloader:
        U_batch, X_batch, Y_batch = U_batch.to(device), X_batch.to(device), Y_batch.to(device)
        segments = find_segments(X_batch[:, :, :state_size], up_thd_scaled, low_thd_scaled, 50)
    
    # 数据保存文件夹
    data_folder = "./data/results/"

    # 运行训练循环
    M = int(3.2e4)  # 定义Epoch数量
    dt = 0.01  # 积分步长
    training_loop(M, state_net, output_net, state_optimizer, output_optimizer,
                  criterion, dataloader, dt, state_size, output_size, data_folder, device, 
                  scalingFactors, sampling_interval, segments)