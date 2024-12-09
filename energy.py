from thop import profile
import torch
def calculate_transmission_energy(data_size, bandwidth, power_transmit):
    """
    计算传输能耗
    :param data_size: 传输数据大小（字节）
    :param bandwidth: 网络带宽（字节/秒）
    :param power_transmit: 传输功率（瓦特）
    :return: 传输能耗（焦耳）
    """
    transmission_time = data_size / bandwidth  # 传输时间（秒）
    energy = power_transmit * transmission_time  # 能耗（焦耳）
    return energy
def calculate_model_size(model):
    """
    计算模型参数的总大小
    :param model: PyTorch 模型
    :return: 模型大小（字节）
    """
    return sum(param.numel() * param.element_size() for param in model.parameters())
def calculate_computation_energy(flops, compute_power, compute_speed):
    """
    计算计算能耗
    :param flops: 执行的浮点运算次数
    :param compute_power: 计算功率（瓦特）
    :param compute_speed: 计算速度（FLOPs/秒）
    :return: 计算能耗（焦耳）
    """
    computation_time = flops / compute_speed  # 计算时间（秒）
    energy = compute_power * computation_time  # 能耗（焦耳）
    return energy
def estimate_flops(model, input_shape):
    """
    使用 thop 库估算模型的 FLOPs
    :param model: PyTorch 模型
    :param input_shape: 输入张量形状
    :return: FLOPs
    """

    dummy_input = torch.randn(*input_shape)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops
