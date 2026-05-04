"""
多 Agent 协作调度器：并行运行预测 Agent 和安全库存 Agent，然后串行执行订货节奏 Agent、仲裁、异常检测。
支持分布式任务队列（Celery）可选。
"""
import concurrent.futures
from typing import Any, Dict

import numpy as np
import pandas as pd

from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.forecasting_agent import ForecastingAgent
from agents.ordering_rhythm_agent import OrderingRhythmAgent
from agents.rule_arbitration import RuleArbitration
from agents.safety_stock_agent import SafetyStockAgent
from data_cleaning import DataCleaner


class OrderingOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        """
        config 示例:
        {
            "forecast_model_type": "lightgbm",
            "service_level": 0.95,
            "lead_time_days": 3,
            "delivery_calendar": {"Monday": True, "Thursday": True},
            "min_pack_multiple": 6,
            "min_turnover_days": 7,
            "floor_coefficient": 0.8,
            "max_inventory_capacity": 1000
        }
        """
        self.config = config
        self.data_cleaner = DataCleaner()
        self.forecast_agent = ForecastingAgent(model_type=config.get("forecast_model_type", "lightgbm"))
        self.safety_agent = SafetyStockAgent(service_level=config.get("service_level", 0.95))
        self.rhythm_agent = OrderingRhythmAgent(
            delivery_calendar=config.get("delivery_calendar", {}),
            min_pack_multiple=config.get("min_pack_multiple", 1),
        )
        self.arbitration = RuleArbitration(
            min_inventory_turnover_days=config.get("min_turnover_days", 7),
            floor_coefficient=config.get("floor_coefficient", 0.8),
        )
        self.anomaly_detector = AnomalyDetectionAgent(
            max_inventory_capacity=config.get("max_inventory_capacity", None),
        )

    def run_for_store_sku(
        self,
        store_sku_id: str,
        historical_sales: list,
        current_inventory: float,
        in_transit: float,
        lead_time_days: int = 3,
    ) -> Dict:
        """
        对单个门店-SKU 运行完整长链推理
        :param store_sku_id: 标识
        :param historical_sales: 历史日销量列表（按时间顺序）
        :param current_inventory: 当前库存
        :param in_transit: 在途库存
        :param lead_time_days: 补货提前期
        :return: 最终订货量及中间结果
        """
        df = pd.DataFrame(
            {
                "date": pd.date_range(end=pd.Timestamp.today(), periods=len(historical_sales)),
                "sales": historical_sales,
            }
        )
        cleaned_df = self.data_cleaner.clean_sales_history(df, date_col="date", sales_col="sales")
        sales_cleaned = cleaned_df["sales"].values

        self.forecast_agent.train(
            history_df=cleaned_df,
            date_col="date",
            sales_col="sales",
            store_sku_id=store_sku_id,
        )
        last_date = cleaned_df["date"].max()
        forecast_daily = self.forecast_agent.predict(
            last_date, history_df=cleaned_df, store_sku_id=store_sku_id
        )

        demand_std = float(np.std(sales_cleaned))
        safety_stock = self.safety_agent.calculate(demand_std, lead_time_days)

        next_delivery_offset = self.rhythm_agent.get_next_delivery_offset(pd.Timestamp.today())
        initial_order = self.rhythm_agent.calculate_order(
            forecast_daily=forecast_daily,
            safety_stock=safety_stock,
            current_inventory=current_inventory,
            in_transit=in_transit,
            lead_time_days=lead_time_days,
            next_delivery_day_offset=next_delivery_offset,
        )

        arbitrated_order = self.arbitration.arbitrate(initial_order, current_inventory, forecast_daily)

        final_order_raw = self.rhythm_agent.apply_pack_multiple(arbitrated_order)

        check_result = self.anomaly_detector.validate_final_order(
            final_order_raw, forecast_daily=forecast_daily
        )

        return {
            "store_sku_id": store_sku_id,
            "forecast_daily": forecast_daily,
            "safety_stock": safety_stock,
            "initial_order": initial_order,
            "arbitrated_order": arbitrated_order,
            "final_order": check_result["final_qty"],
            "approved": check_result["approved"],
            "reason": check_result["reason"],
        }

    def run_batch(self, store_sku_list: list, parallel=True, max_workers=4):
        """
        批量处理多个门店-SKU，支持并行。每个任务使用独立 Orchestrator，避免 Agent 状态并发冲突。
        """
        results = []

        def run_one(item: dict):
            orch = OrderingOrchestrator(self.config)
            return orch.run_for_store_sku(**item)

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_one, item) for item in store_sku_list]
                for fut in concurrent.futures.as_completed(futures):
                    results.append(fut.result())
        else:
            for item in store_sku_list:
                results.append(run_one(item))
        return results
