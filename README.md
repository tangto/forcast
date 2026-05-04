# 智能多Agent订货系统

实现长链推理与多Agent协作的库存优化系统。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 运行示例：`python main.py`

## 模块说明

- `data_cleaning.py`: 数据清洗
- `agents/`: 包含预测、安全库存、订货节奏、异常检测、规则仲裁
- `orchestrator.py`: 多Agent协作调度
- `main.py`: 演示程序

## 系统特点

- 长链推理（6步）
- 并行预测与安全库存计算
- 规则仲裁防止过度压货
- 异常检测防幻觉
