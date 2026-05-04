"""
安全库存 Agent：基于需求标准差、补货提前期、目标服务水平（95%）计算每日安全库存。
"""
import numpy as np
from scipy.stats import norm


class SafetyStockAgent:
    def __init__(self, service_level=0.95):
        """
        :param service_level: 目标服务水平，默认 0.95 (95%)
        """
        self.service_level = service_level
        self.z_score = norm.ppf(service_level)

    def calculate(self, demand_std: float, lead_time_days: float) -> float:
        """
        计算安全库存
        :param demand_std: 日需求标准差（历史数据计算得出）
        :param lead_time_days: 补货提前期（天）
        :return: 安全库存量
        """
        safety_stock = self.z_score * demand_std * np.sqrt(lead_time_days)
        return round(float(safety_stock), 2)

    def calculate_for_store_sku(self, history_series: np.ndarray, lead_time_days: float) -> float:
        """
        根据历史销量序列直接计算安全库存
        :param history_series: 历史每日销量（一维数组）
        :param lead_time_days: 补货提前期（天）
        :return: 安全库存量
        """
        demand_std = float(np.std(history_series))
        return self.calculate(demand_std, lead_time_days)
