"""
订货节奏 Agent：综合预测需求量、安全库存、当前库存、在途库存，按门店配送日历和最小包装倍数输出最终订货量。
"""
import math


class OrderingRhythmAgent:
    def __init__(self, delivery_calendar, min_pack_multiple=1):
        """
        :param delivery_calendar: 配送日历，例如 {"Monday": True, "Thursday": True} 表示周一和周四配送
        :param min_pack_multiple: 最小包装倍数（订货量须为此数的整数倍）
        """
        self.delivery_calendar = delivery_calendar
        self.min_pack_multiple = min_pack_multiple

    def calculate_order(
        self,
        forecast_daily: float,
        safety_stock: float,
        current_inventory: float,
        in_transit: float,
        lead_time_days: int,
        next_delivery_day_offset: int = 1,
    ) -> float:
        """
        计算初步订货量
        :param forecast_daily: 预测日需求量
        :param safety_stock: 安全库存量
        :param current_inventory: 当前库存
        :param in_transit: 在途库存（已下单未到货）
        :param lead_time_days: 补货提前期（天）
        :param next_delivery_day_offset: 距离下次配送的天数（用于覆盖至到货后的需求）
        :return: 初步订货量（未取整）
        """
        days_to_cover = lead_time_days + next_delivery_day_offset
        required_qty = forecast_daily * days_to_cover + safety_stock
        available = current_inventory + in_transit
        order_qty = max(0, required_qty - available)
        return order_qty

    def apply_pack_multiple(self, order_qty: float) -> int:
        """按最小包装倍数取整（向上取整）"""
        if self.min_pack_multiple <= 1:
            return int(math.ceil(order_qty))
        return int(math.ceil(order_qty / self.min_pack_multiple)) * self.min_pack_multiple

    def get_next_delivery_offset(self, today):
        """根据配送日历计算距离下一次配送的天数（业务逻辑示例）"""
        # 简化：假设今天之后第一个配送日
        days_ahead = 1
        return days_ahead
