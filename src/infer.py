import torch
import scipy.io
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

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
        # 将3个可能的驾驶模式变量映射到5维空间
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
    matrixData_t = MatlabData['matrixData_t'][0]
    matrixData_v = MatlabData['matrixData_v'][0]
    scalingFactors = MatlabData['scalingFactors']

    # 根据指定的列索引选择列
    # ActualGear | ActTq_NM | MCPressure_Bar | SteerAngle_deg | LongitudinalVelocity_kph | YawRate_deg_sec | 
    # LongitudinalAcceleration_g | LateralAcceleration_g
    CsvData_t_array = [x[:, [3, 20, 21, 22, 8, 9, 18, 19]] for x in matrixData_t]
    CsvData_v_array = [x[:, [3, 20, 21, 22, 8, 9, 18, 19]] for x in matrixData_v]

    return CsvData_t_array, CsvData_v_array, scalingFactors

def normalize(CsvData_array, scalingFactors):
# 初始化CsvData_norm为和CsvData_array相同结构的空列表
    CsvData_norm = []
    # 对CsvData_array中的每个矩阵进行处理

    u1_scale = scalingFactors[0][0]['ActTq_NM']['scale'][0, 0][0, 0]
    u1_offset = scalingFactors[0][0]['ActTq_NM']['offset'][0, 0][0, 0]
    u2_scale = scalingFactors[0][0]['MCPressure_Bar']['scale'][0, 0][0, 0]
    u2_offset = scalingFactors[0][0]['MCPressure_Bar']['offset'][0, 0][0, 0]
    u3_scale = scalingFactors[0][0]['SteerAngle_deg']['scale'][0, 0][0, 0]
    u3_offset = scalingFactors[0][0]['SteerAngle_deg']['offset'][0, 0][0, 0]

    x1_scale = scalingFactors[0][0]['LongitudinalVelocity_kph']['scale'][0, 0][0, 0]
    x1_offset = scalingFactors[0][0]['LongitudinalVelocity_kph']['offset'][0, 0][0, 0]
    x2_scale = scalingFactors[0][0]['YawRate_deg_sec']['scale'][0, 0][0, 0]
    x2_offset = scalingFactors[0][0]['YawRate_deg_sec']['offset'][0, 0][0, 0]

    y1_scale = scalingFactors[0][0]['LongitudinalAcceleration_g']['scale'][0, 0][0, 0]
    y1_offset = scalingFactors[0][0]['LongitudinalAcceleration_g']['offset'][0, 0][0, 0]
    y2_scale = scalingFactors[0][0]['LateralAcceleration_g']['scale'][0, 0][0, 0]
    y2_offset = scalingFactors[0][0]['LateralAcceleration_g']['offset'][0, 0][0, 0]

    for matrix in CsvData_array:
        # 对每列应用缩放和偏移
        norm_matrix = matrix.copy()  # 复制原矩阵以保留原始数据
        norm_matrix[:, 0] = matrix[:, 0]
        norm_matrix[:, 1] = matrix[:, 1] * u1_scale + u1_offset
        norm_matrix[:, 2] = matrix[:, 2] * u2_scale + u2_offset
        norm_matrix[:, 3] = matrix[:, 3] * u3_scale + u3_offset
        norm_matrix[:, 4] = matrix[:, 4] * x1_scale + x1_offset
        norm_matrix[:, 5] = matrix[:, 5] * x2_scale + x2_offset
        norm_matrix[:, 6] = matrix[:, 6] * y1_scale + y1_offset
        norm_matrix[:, 7] = matrix[:, 7] * y2_scale + y2_offset

        # 将处理后的矩阵添加到CsvData_norm中
        CsvData_norm.append(norm_matrix)

    # 初始化U, X, Y, U_norm, X_norm, Y_norm为空列表
    U, X, Y = [], [], []
    U_norm, X_norm, Y_norm = [], [], []

    # 遍历CsvData_array和CsvData_norm，提取指定列
    for matrix, norm_matrix in zip(CsvData_array, CsvData_norm):
        # 从原始数据中提取
        U.append(matrix[:, [0, 1, 2, 3]])   # 前4列
        X.append(matrix[:, [4, 5]])         # 第5、6列
        Y.append(matrix[:, [6, 7]])         # 后2列

        # 从标准化数据中提取
        U_norm.append(norm_matrix[:, [0, 1, 2, 3]]) # 前4列
        X_norm.append(norm_matrix[:, [4, 5]])       # 第5、6列
        Y_norm.append(norm_matrix[:, [6, 7]])       # 后2列

    return CsvData_norm, U, X, Y, U_norm, X_norm, Y_norm

def calculate_next_state(state_net, last_state, u_tensor, scalingFactors, dt):
    
    u1_scale = scalingFactors[0][0]['ActTq_NM']['scale'][0, 0][0, 0]
    u1_offset = scalingFactors[0][0]['ActTq_NM']['offset'][0, 0][0, 0]
    u2_scale = scalingFactors[0][0]['MCPressure_Bar']['scale'][0, 0][0, 0]
    u2_offset = scalingFactors[0][0]['MCPressure_Bar']['offset'][0, 0][0, 0]
    u3_scale = scalingFactors[0][0]['SteerAngle_deg']['scale'][0, 0][0, 0]
    u3_offset = scalingFactors[0][0]['SteerAngle_deg']['offset'][0, 0][0, 0]
    
    x1_scale = scalingFactors[0][0]['LongitudinalVelocity_kph']['scale'][0, 0][0, 0]
    x1_offset = scalingFactors[0][0]['LongitudinalVelocity_kph']['offset'][0, 0][0, 0]
    x2_scale = scalingFactors[0][0]['YawRate_deg_sec']['scale'][0, 0][0, 0]
    x2_offset = scalingFactors[0][0]['YawRate_deg_sec']['offset'][0, 0][0, 0]

    u = u_tensor[:, :]  # 选取u_tensor中的全部四个元素
    state_input = torch.cat((last_state.unsqueeze(0), u), dim=1)
    dxdt = state_net(state_input)
    next_states_unclamped = last_state + dxdt.squeeze(0) * dt
    next_states_clamped = next_states_unclamped.clone()
    x1_zero_scaled = 0.0 * x1_scale + x1_offset
    if u_tensor[0][0] > 0.5:
        next_states_clamped[0] = torch.clamp(next_states_unclamped[0], min=x1_zero_scaled)
        x2_est = ((next_states_clamped[0] - x1_offset) / x1_scale / 3.6) / 3.0 * torch.tan(((u[:, 3].squeeze(0) - u3_offset) / u3_scale) / 180.0 * torch.pi / 13.0)/torch.pi*180
        x2_1_scaled = x2_est * 1.2 * x2_scale + x2_offset
        x2_2_scaled = x2_est * 0.8 * x2_scale + x2_offset
        x2_high_scaled = torch.max(x2_1_scaled, x2_2_scaled)
        x2_low_scaled = torch.min(x2_1_scaled, x2_2_scaled)
        next_states = next_states_clamped.clone()
        next_states[1] = torch.clamp(next_states_clamped[1], min=x2_low_scaled, max=x2_high_scaled)
    elif u_tensor[0][0] < -0.5:
        next_states_clamped[0] = torch.clamp(next_states_unclamped[0], max=x1_zero_scaled)
        x2_est = ((next_states_clamped[0] - x1_offset) / x1_scale / 3.6) / 3.0 * torch.tan(((u[:, 3].squeeze(0) - u3_offset) / u3_scale) / 180.0 * torch.pi / 13.0)/torch.pi*180
        x2_1_scaled = x2_est * 1.2 * x2_scale + x2_offset
        x2_2_scaled = x2_est * 0.8 * x2_scale + x2_offset
        x2_high_scaled = torch.max(x2_1_scaled, x2_2_scaled)
        x2_low_scaled = torch.min(x2_1_scaled, x2_2_scaled)
        next_states = next_states_clamped.clone()
        next_states[1] = torch.clamp(next_states_clamped[1], min=x2_low_scaled, max=x2_high_scaled)
    else:
        next_state_ = torch.zeros_like(last_state)
        next_states = next_state_.clone()
        next_states[0] = next_state_[0] * x1_scale + x1_offset
        next_states[1] = next_state_[1] * x2_scale + x2_offset

    return next_states

# 定义一个函数来绘制和保存图表
def plot_and_save(data1, data2, data3, data4, sequence_number, data_folder, nameX):
    time = np.linspace(0, (len(data1) - 1) * dt, len(data1))

    # 启用LaTeX渲染器和设置字体
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 9

    fig, axs = plt.subplots(4, 1, figsize=(3.5, 4))  # 尺寸转换为英寸

    # 设置每个子图的 X 轴范围
    x_min, x_max = 0, (len(data1) - 1) * dt

    # 设置 Y 轴标签的统一位置
    y_label_x_position = -0.15

    # 设置X轴只显示整数刻度
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 第一个图
    axs[0].plot(time, data1[:, 0], 'b-', label='Experiment', linewidth=0.5)
    axs[0].plot(time, data2[:, 0], 'r-', label='Simulation', linewidth=0.5)
    axs[0].set_ylabel('$V_x$ (km/h)')
    axs[0].set_xlim(x_min, x_max)
    axs[0].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[0].set_xticklabels([])

    # 第二个图
    axs[1].plot(time, data1[:, 1], 'b-', label='Experiment', linewidth=0.5)
    axs[1].plot(time, data2[:, 1], 'r-', label='Simulation', linewidth=0.5)
    axs[1].set_ylabel('$\dot{\psi}$ (deg/sec)')
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[1].set_xlim(x_min, x_max)
    axs[1].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[1].set_xticklabels([])
    
    # 第三个图
    axs[2].plot(time, data3[:, 0], 'b-', label='Experiment', linewidth=0.5)
    axs[2].plot(time, data4[:, 0], 'r-', label='Simulation', linewidth=0.5)
    axs[2].set_ylabel('$a_x$ ($g$)')
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].set_xlim(x_min, x_max)
    axs[2].yaxis.set_label_coords(y_label_x_position, 0.5)
    axs[2].set_xticklabels([])

    # 第四个图
    axs[3].plot(time, data3[:, 1], 'b-', label='Experiment', linewidth=0.5)
    axs[3].plot(time, data4[:, 1], 'r-', label='Simulation', linewidth=0.5)
    axs[3].set_xlabel('Time (sec)')
    axs[3].set_ylabel('$a_y$ ($g$)')
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[3].set_xlim(x_min, x_max)
    axs[3].yaxis.set_label_coords(y_label_x_position, 0.5)

    # 设置图例
    fig.legend(['Experiment', 'Simulation'], loc='upper center', ncol=2, bbox_to_anchor=(0.58, 1))

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)  # 调整顶部和底部边距

    # 保存图表
    plt.savefig(f"{data_folder}/{nameX}_{sequence_number}.png", dpi=600)
    plt.close()

def calculate_errors(actual_list, predicted_list, variable_index):
    # Concatenate all sequences for each variable
    actual_var = np.concatenate([seq[:, variable_index] for seq in actual_list])
    predicted_var = np.concatenate([seq[:, variable_index] for seq in predicted_list])

    # Calculate MSE and MAE for the concatenated sequences
    mse = mean_squared_error(actual_var, predicted_var)
    mae = mean_absolute_error(actual_var, predicted_var)
    
    return mse, mae

# 主程序入口点
if __name__ == "__main__":
    # 定义网络结构并创建StateNetwork的实例
    input_size, state_size, output_size = 4, 2, 2
    state_net = CustomNetwork([input_size + state_size] + [64] * 6 + [state_size])
    output_net = CustomNetwork([input_size + state_size - 1] + [64] * 3 + [output_size])

    # 设定DATA路径
    data_folder = "./data/results/"

    # 加载训练好的模型参数
    state_net.load_state_dict(torch.load(Path(data_folder) / f'state_net_32000.pth'))
    output_net.load_state_dict(torch.load(Path(data_folder) / f'output_net_32000.pth'))

    # 在推理时不需要梯度
    state_net.eval()
    output_net.eval()

    CsvData_t_array, CsvData_v_array, scalingFactors = load_data('./data/MatlabData.mat')

    CsvData_t_norm, U_t, X_t, Y_t, U_t_norm, X_t_norm, Y_t_norm = normalize(CsvData_t_array, scalingFactors)

    # 定义积分步长
    dt = 0.01

    U_t_torch = [torch.tensor(U_t_norm[i], dtype=torch.float32) for i in range(len(U_t_norm))]

    # 初始化X_pred_torch为空列表
    X_pred_t_torch = []

    for i in range(len(U_t_torch)):
        # 初始状态，添加到states_pred列表中
        initial_state = torch.tensor(X_t_norm[i][0,:], dtype=torch.float32).unsqueeze(0)
        states_pred = [initial_state]  

        for t in range(1, U_t_torch[i].shape[0]):
            last_state = states_pred[-1].squeeze(0)  # 确保last_state是1D张量
            u_tensor = U_t_torch[i][t-1].unsqueeze(0)  # 将u_tensor转换为2D张量
            next_state = calculate_next_state(state_net, last_state, u_tensor, scalingFactors, dt)
            states_pred.append(next_state.unsqueeze(0))

        # 使用整个states_pred列表来创建states_pred_stacked
        states_pred_stacked = torch.cat(states_pred, dim=0)
        X_pred_t_torch.append(states_pred_stacked)


    x1_scale = scalingFactors[0][0]['LongitudinalVelocity_kph']['scale'][0, 0][0, 0]
    x1_offset = scalingFactors[0][0]['LongitudinalVelocity_kph']['offset'][0, 0][0, 0]
    x2_scale = scalingFactors[0][0]['YawRate_deg_sec']['scale'][0, 0][0, 0]
    x2_offset = scalingFactors[0][0]['YawRate_deg_sec']['offset'][0, 0][0, 0]

    # 将X_pred_torch中的数据转换为NumPy数组，并保存在X_pred_norm中
    X_pred_t_norm = [x.detach().numpy() for x in X_pred_t_torch]

    X_pred_t = []

    for matrix in X_pred_t_norm:
        denorm_matrix = matrix.copy()  # 复制原矩阵以保留原始数据
        denorm_matrix[:, 0] = ( matrix[:, 0] - x1_offset ) / x1_scale 
        denorm_matrix[:, 1] = ( matrix[:, 1] - x2_offset ) / x2_scale

        # 将处理后的矩阵添加到CsvData_norm中
        X_pred_t.append(denorm_matrix)

    y1_scale = scalingFactors[0][0]['LongitudinalAcceleration_g']['scale'][0, 0][0, 0]
    y1_offset = scalingFactors[0][0]['LongitudinalAcceleration_g']['offset'][0, 0][0, 0]
    y2_scale = scalingFactors[0][0]['LateralAcceleration_g']['scale'][0, 0][0, 0]
    y2_offset = scalingFactors[0][0]['LateralAcceleration_g']['offset'][0, 0][0, 0]

    # 初始化输出预测列表
    Y_pred_t_torch = []

    for i in range(len(U_t_torch)):
        y_pred = []  # 初始化单个序列的预测列表
        for t in range(U_t_torch[i].shape[0]):
            state_input = X_pred_t_torch[i][t,:].unsqueeze(0)  # 获取当前状态
            u_input = U_t_torch[i][t][1:].unsqueeze(0)  # 获取当前控制输入（去掉档位信息）
            # 合并状态和控制输入
            combined_input = torch.cat((state_input, u_input), dim=1)
            y_pred_output = output_net(combined_input).squeeze(0)
            y_pred.append(y_pred_output)

        # 将单个序列的预测添加到总预测列表中
        Y_pred_t_torch.append(torch.stack(y_pred))

    Y_pred_t_norm = [y.detach().numpy() for y in Y_pred_t_torch]

    Y_pred_t = []

    for matrix in Y_pred_t_norm:
        denorm_matrix = matrix.copy() 
        denorm_matrix[:, 0] = ( matrix[:, 0] - y1_offset ) / y1_scale 
        denorm_matrix[:, 1] = ( matrix[:, 1] - y2_offset ) / y2_scale
        Y_pred_t.append(denorm_matrix)

    # 对于X和X_pred、Y和Y_pred的每个序列绘制和保存图表
    for i in range(len(X_t)):
        plot_and_save(X_t[i], X_pred_t[i], Y_t[i], Y_pred_t[i], i, data_folder, 'training')

    # Define variable indices corresponding to Vx, Yaw, Ax, Ay
    variable_indices = {
        'Vx_t': 0,
        'Yaw_t': 1,
        'Ax_t': 0,
        'Ay_t': 1,
    }

    # Calculate MSE and MAE for each variable
    for var_name, var_index in variable_indices.items():
        if var_name in ['Vx_t', 'Yaw_t']:
            mse, mae = calculate_errors(X_t, X_pred_t, var_index)
            mse_norm, mae_norm = calculate_errors(X_t_norm, X_pred_t_norm, var_index)
        else:  # for 'Ax', 'Ay'
            mse, mae = calculate_errors(Y_t, Y_pred_t, var_index)
            mse_norm, mae_norm = calculate_errors(Y_t_norm, Y_pred_t_norm, var_index)

        print(f'MSE of {var_name}: {mse}')
        print(f'MAE of {var_name}: {mae}')
        
        print(f'MSE of {var_name} (scaled): {mse_norm}')
        print(f'MAE of {var_name} (scaled): {mae_norm}')

    #############################################################################################
    print()
    #############################################################################################

    CsvData_v_norm, U_v, X_v, Y_v, U_v_norm, X_v_norm, Y_v_norm = normalize(CsvData_v_array, scalingFactors)

    U_v_torch = [torch.tensor(U_v_norm[i], dtype=torch.float32) for i in range(len(U_v_norm))]

    X_pred_v_torch = []

    for i in range(len(U_v_torch)):
        initial_state = torch.tensor(X_v_norm[i][0,:], dtype=torch.float32).unsqueeze(0)
        states_pred = [initial_state]  

        for t in range(1, U_v_torch[i].shape[0]):
            last_state = states_pred[-1].squeeze(0)
            u_tensor = U_v_torch[i][t-1].unsqueeze(0)
            next_state = calculate_next_state(state_net, last_state, u_tensor, scalingFactors, dt)
            states_pred.append(next_state.unsqueeze(0))
            
        states_pred_stacked = torch.cat(states_pred, dim=0)
        X_pred_v_torch.append(states_pred_stacked)
        
    X_pred_v_norm = [x.detach().numpy() for x in X_pred_v_torch]

    X_pred_v = []

    for matrix in X_pred_v_norm:
        denorm_matrix = matrix.copy()
        denorm_matrix[:, 0] = ( matrix[:, 0] - x1_offset ) / x1_scale 
        denorm_matrix[:, 1] = ( matrix[:, 1] - x2_offset ) / x2_scale
        
        X_pred_v.append(denorm_matrix)
        
    Y_pred_v_torch = []

    for i in range(len(U_v_torch)):
        y_pred = []
        for t in range(U_v_torch[i].shape[0]):
            state_input = X_pred_v_torch[i][t,:].unsqueeze(0)
            u_input = U_v_torch[i][t][1:].unsqueeze(0)
            
            combined_input = torch.cat((state_input, u_input), dim=1)
            y_pred_output = output_net(combined_input).squeeze(0)
            y_pred.append(y_pred_output)
            
        Y_pred_v_torch.append(torch.stack(y_pred))

    Y_pred_v_norm = [y.detach().numpy() for y in Y_pred_v_torch]

    Y_pred_v = []

    for matrix in Y_pred_v_norm:
        denorm_matrix = matrix.copy() 
        denorm_matrix[:, 0] = ( matrix[:, 0] - y1_offset ) / y1_scale 
        denorm_matrix[:, 1] = ( matrix[:, 1] - y2_offset ) / y2_scale
        Y_pred_v.append(denorm_matrix)

    for i in range(len(X_v)):
        plot_and_save(X_v[i], X_pred_v[i], Y_v[i], Y_pred_v[i], i, data_folder, 'validation')
        
    variable_indices = {
        'Vx_v': 0,
        'Yaw_v': 1,
        'Ax_v': 0,
        'Ay_v': 1,
    }
    
    for var_name, var_index in variable_indices.items():
        if var_name in ['Vx_v', 'Yaw_v']:
            mse, mae = calculate_errors(X_v, X_pred_v, var_index)
            mse_norm, mae_norm = calculate_errors(X_v_norm, X_pred_v_norm, var_index)
        else:
            mse, mae = calculate_errors(Y_v, Y_pred_v, var_index)
            mse_norm, mae_norm = calculate_errors(Y_v_norm, Y_pred_v_norm, var_index)

        print(f'MSE of {var_name}: {mse}')
        print(f'MAE of {var_name}: {mae}')
        
        print(f'MSE of {var_name} (scaled): {mse_norm}')
        print(f'MAE of {var_name} (scaled): {mae_norm}')