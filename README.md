# ai2026
大作业

运行入口保持不变：

```bash
python deep_learning_stock_prediction.py
python mlp_baseline.py
```

代码已拆分到 `stock_prediction/`：

- `settings.py`：路径、数据列类型、MLP/GRU 配置
- `data.py`：基础信息、交易日历、日频数据、股票池和基准数据加载
- `features.py`：技术指标、标签构造、滚动标准化
- `sequences.py`：MLP/GRU 样本序列构造和标签处理
- `models.py`：Dataset、MLP、GRU 模型
- `training.py`：训练循环和 loss 曲线输出
- `evaluation.py`：IC、ICIR、方向胜率和预测工具
- `backtest.py`：价格透视表、回测、指标和曲线
- `signals.py`：比赛交易信号生成和汇总
- `pipelines/`：两个实验的主流程编排
