"""
数据清洗模块：去除异常值、处理缺失值、剔除促销期影响。
"""
import pandas as pd


class DataCleaner:
    def __init__(self, outlier_method="iqr", iqr_multiplier=3.0):
        """
        :param outlier_method: 'iqr' 或 'zscore'
        :param iqr_multiplier: IQR 倍数，默认3
        """
        self.outlier_method = outlier_method
        self.iqr_multiplier = iqr_multiplier

    def clean_sales_history(
        self,
        df: pd.DataFrame,
        date_col="date",
        sales_col="sales",
        promo_col=None,
        promo_exclude_values=None,
    ) -> pd.DataFrame:
        """
        清洗历史销售数据
        :param df: 原始数据，必须包含日期和销量列
        :param date_col: 日期列名
        :param sales_col: 销量列名
        :param promo_col: 促销标记列名（如果存在，可排除促销期数据）
        :param promo_exclude_values: 需要排除的促销标记值列表，例如 [1, 'promo']
        :return: 清洗后的 DataFrame
        """
        df = df.copy()
        # 确保日期类型并排序
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # 缺失按 0 处理（可按业务改为 ffill）
        df[sales_col] = df[sales_col].fillna(0)

        if promo_col is not None and promo_exclude_values is not None:
            mask = ~df[promo_col].isin(promo_exclude_values)
            df = df[mask].copy()

        if self.outlier_method == "iqr":
            q1 = df[sales_col].quantile(0.25)
            q3 = df[sales_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            df = df[(df[sales_col] >= lower_bound) & (df[sales_col] <= upper_bound)]
        elif self.outlier_method == "zscore":
            mean = df[sales_col].mean()
            std = df[sales_col].std()
            df = df[(df[sales_col] - mean).abs() <= self.iqr_multiplier * std]

        return df
