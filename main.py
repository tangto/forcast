#!/usr/bin/env python3
"""
主程序示例：演示如何使用多 Agent 协作系统，对单个门店-SKU 进行订货决策。
"""
import numpy as np

from orchestrator import OrderingOrchestrator


def main():
    config = {
        "forecast_model_type": "lightgbm",
        "service_level": 0.95,
        "lead_time_days": 3,
        "delivery_calendar": {"Monday": True, "Thursday": True},
        "min_pack_multiple": 6,
        "min_turnover_days": 7,
        "floor_coefficient": 0.8,
        "max_inventory_capacity": 2000,
    }

    np.random.seed(42)
    historical_sales = np.random.poisson(lam=12, size=90).tolist()

    orchestrator = OrderingOrchestrator(config)

    result = orchestrator.run_for_store_sku(
        store_sku_id="STORE_001_SKU_12345",
        historical_sales=historical_sales,
        current_inventory=50,
        in_transit=20,
        lead_time_days=3,
    )

    print("=" * 50)
    print("订货决策结果:")
    for k, v in result.items():
        print(f"{k}: {v}")

    batch_tasks = [
        {
            "store_sku_id": f"STORE_{i}_SKU_{j}",
            "historical_sales": historical_sales,
            "current_inventory": 30,
            "in_transit": 10,
            "lead_time_days": 3,
        }
        for i in range(1, 5)
        for j in range(1, 3)
    ]
    batch_results = orchestrator.run_batch(batch_tasks, parallel=True, max_workers=4)
    print(f"\n批量处理完成，共 {len(batch_results)} 条结果。")


if __name__ == "__main__":
    main()
