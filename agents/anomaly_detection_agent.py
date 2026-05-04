"""
异常检测 Agent：校验最终订货量是否超出合理范围（历史3σ阈值、最大库存容量等），防止幻觉。
"""
import numpy as np


class AnomalyDetectionAgent:
    def __init__(self, max_inventory_capacity=None, outlier_threshold=3.0):
        """
        :param max_inventory_capacity: 仓库最大库存容量（可选）
        :param outlier_threshold: 用于识别相对于历史订货量的异常倍数（标准差倍数）
        """
        self.max_capacity = max_inventory_capacity
        self.outlier_threshold = outlier_threshold
        self.history_order_qties = []

    def add_history(self, order_qty: float):
        self.history_order_qties.append(order_qty)

    def check(self, order_qty: float, context: dict = None) -> tuple:
        """
        校验订货量是否异常
        :param order_qty: 待校验的订货量
        :param context: 上下文信息，例如预测值、安全库存等，用于更精细检查
        :return: (是否通过, 原因字符串)
        """
        if order_qty < 0:
            return False, f"订货量为负数: {order_qty}"

        if self.max_capacity and order_qty > self.max_capacity:
            return False, f"订货量 {order_qty} 超过最大库存容量 {self.max_capacity}"

        if len(self.history_order_qties) >= 5:
            mean = np.mean(self.history_order_qties)
            std = np.std(self.history_order_qties)
            if std > 0:
                z = (order_qty - mean) / std
                if abs(z) > self.outlier_threshold:
                    return False, f"订货量异常偏离历史均值 (z={z:.2f})"

        if context and "forecast_daily" in context:
            forecast = context["forecast_daily"]
            if order_qty > forecast * 10:
                return False, f"订货量 {order_qty} 远超预测值 {forecast}"

        return True, "校验通过"

    def validate_final_order(self, order_qty: float, forecast_daily: float = None) -> dict:
        """友好接口"""
        context = {}
        if forecast_daily:
            context["forecast_daily"] = forecast_daily
        approved, reason = self.check(order_qty, context)
        return {"approved": approved, "reason": reason, "final_qty": order_qty if approved else 0}
