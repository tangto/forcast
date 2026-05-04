"""
预测 Agent：基于 LightGBM 或 Prophet 模型，对每个门店-SKU 预测未来 T+14 天的日均需求量。
"""
import os
import pickle

import numpy as np
import pandas as pd


class ForecastingAgent:
    def __init__(self, model_type="lightgbm", horizon=14, retrain_frequency="daily"):
        """
        :param model_type: 'lightgbm' 或 'prophet'
        :param horizon: 预测天数，默认14
        :param retrain_frequency: 重训练频率，'daily' 或 'weekly'
        """
        self.model_type = model_type
        self.horizon = horizon
        self.retrain_frequency = retrain_frequency
        self.model = None

    def train(
        self,
        history_df: pd.DataFrame,
        date_col="date",
        sales_col="sales",
        features=None,
        store_sku_id=None,
    ):
        """
        训练模型（按门店-SKU 单独训练或全局特征）
        :param history_df: 历史数据，包含日期和销量
        :param date_col: 日期列名
        :param sales_col: 销量列名
        :param features: 额外特征列名列表（如星期几、节假日等）
        :param store_sku_id: 当前门店-SKU 标识，用于保存模型
        """
        df = history_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        if self.model_type == "lightgbm":
            from lightgbm import LGBMRegressor

            df["day_of_week"] = df[date_col].dt.dayofweek
            df["month"] = df[date_col].dt.month
            df["lag1"] = df[sales_col].shift(1)
            df["lag7"] = df[sales_col].shift(7)
            df["rolling_mean_7"] = df[sales_col].rolling(7, min_periods=1).mean()
            df = df.dropna().reset_index(drop=True)

            feature_cols = ["day_of_week", "month", "lag1", "lag7", "rolling_mean_7"]
            if features:
                feature_cols += features
            x = df[feature_cols]
            y = df[sales_col]

            self.model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1,
            )
            self.model.fit(x, y)

        elif self.model_type == "prophet":
            from prophet import Prophet

            df_prophet = df.rename(columns={date_col: "ds", sales_col: "y"})
            self.model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            self.model.fit(df_prophet)

        if store_sku_id:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/forecast_{store_sku_id}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

    def predict(self, last_date, history_df=None, store_sku_id=None) -> float:
        """
        预测未来 horizon 天的日均需求量
        :param last_date: 历史数据的最后日期（datetime 对象）
        :param history_df: LightGBM 分支需要最近销量构造特征
        :param store_sku_id: 用于加载已保存的模型
        :return: 预测日均需求量（浮点数）
        """
        if store_sku_id and self.model is None:
            model_path = f"models/forecast_{store_sku_id}.pkl"
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

        if self.model is None:
            raise ValueError("模型未训练或未加载，请先调用 train() 方法")

        if self.model_type == "lightgbm":
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.horizon)
            preds = []
            for date in future_dates:
                features = {
                    "day_of_week": date.dayofweek,
                    "month": date.month,
                    "lag1": history_df["sales"].iloc[-1] if history_df is not None else 0,
                    "lag7": history_df["sales"].iloc[-7:].mean() if history_df is not None else 0,
                    "rolling_mean_7": history_df["sales"].iloc[-7:].mean() if history_df is not None else 0,
                }
                preds.append(self.model.predict(pd.DataFrame([features]))[0])
            return float(np.mean(preds))

        if self.model_type == "prophet":
            future = self.model.make_future_dataframe(periods=self.horizon, include_history=False)
            forecast = self.model.predict(future)
            preds = forecast["yhat"].values
            return float(np.mean(preds))

        raise ValueError(f"未知 model_type: {self.model_type}")
