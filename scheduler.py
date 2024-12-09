class Scheduler:
    """
    调度系统，负责用户调度逻辑
    """
    def __init__(self, tau_start, tau_end, total_steps):
        self.tau_start = tau_start  # 初始温度
        self.tau_end = tau_end  # 最低温度
        self.total_steps = total_steps  # 总时间步
        self.current_step = 0  # 当前时间步
        self.tau=self.get_tau()

    def get_tau(self):
        """
        动态调整温度参数
        :return: 当前温度
        """
        tau = max(self.tau_end, self.tau_start - (self.tau_start - self.tau_end) * (self.current_step / self.total_steps))
        self.current_step += 1
        return tau