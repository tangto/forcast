"""
规则仲裁模块：当预测+安全库存导致库存周转天数下降过快时，强制执行保底系数。
"""


class RuleArbitration:
    def __init__(self, min_inventory_turnover_days=7, floor_coefficient=0.8):
        """
        :param min_inventory_turnover_days: 最低库存周转天数红线
        :param floor_coefficient: 保底系数（例如0.8表示按计算值的80%订货）
        """
        self.min_turnover_days = min_inventory_turnover_days
        self.floor_coeff = floor_coefficient

    def arbitrate(self, initial_order_qty: float, current_inventory: float, forecast_daily: float) -> float:
        """
        根据库存周转天数调整订货量
        :param initial_order_qty: 初始计算出的订货量
        :param current_inventory: 当前库存
        :param forecast_daily: 预测日需求
        :return: 调整后的订货量
        """
        if forecast_daily <= 0:
            return initial_order_qty

        future_inventory = current_inventory + initial_order_qty
        turnover_days = future_inventory / forecast_daily

        if turnover_days < self.min_turnover_days:
            adjusted = initial_order_qty * self.floor_coeff
            print(
                f"[仲裁] 库存周转天数 {turnover_days:.1f} < {self.min_turnover_days}，"
                f"应用保底系数 {self.floor_coeff}，订货量 {initial_order_qty} -> {adjusted:.1f}"
            )
            return max(0, adjusted)
        return initial_order_qty
