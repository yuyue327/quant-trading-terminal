import os

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# 使用绝对路径（请根据你的实际路径调整）
BASE_DIR = "/Users/mac/Desktop/ACML"
DATA_DIR = os.path.join(BASE_DIR, "data", "results")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")


def load_summary() -> Dict[str, Any]:
    """加载多股票汇总结果"""
    path = os.path.join(DATA_DIR, "adaptive_repeat_summary.csv")
    if not os.path.exists(path):
        return {"error": "Summary not found, please run step49 first"}
    df = pd.read_csv(path)
    df = df.fillna(0)
    return df.to_dict(orient="records")


def load_stock_probs(stock: str) -> List[Dict[str, Any]]:
    """加载单只股票的预测概率序列"""
    # 尝试多种文件名格式
    possible_names = [
        f"adaptive_probs_{stock}.csv",
        f"adaptive_probs_{stock.replace('_', '.')}.csv",
    ]
    path = None
    for name in possible_names:
        test_path = os.path.join(DATA_DIR, name)
        if os.path.exists(test_path):
            path = test_path
            break
    # 如果找不到，尝试模糊匹配
    if path is None:
        for f in os.listdir(DATA_DIR):
            if f.startswith("adaptive_probs_") and stock in f:
                path = os.path.join(DATA_DIR, f)
                break
    if path is None or not os.path.exists(path):
        return []
    df = pd.read_csv(path, parse_dates=[0])
    df = df.fillna(0)
    records = []
    for _, row in df.iterrows():
        date_val = row.iloc[0]
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
        records.append({
            "date": date_str,
            "prob": float(row.get('prob', 0.5)),
            "uncertainty": float(row.get('uncertainty', 0))
        })
    return records


def load_ohlc(stock: str) -> List[Dict[str, Any]]:
    """加载原始 OHLC 数据（支持中文名模糊匹配）"""
    # 先尝试精确匹配
    possible_names = [
        f"{stock}.parquet",
        f"{stock.replace('_', '.')}.parquet",
    ]
    path = None
    for name in possible_names:
        test_path = os.path.join(FEATURES_DIR, name)
        if os.path.exists(test_path):
            path = test_path
            break

    if path is None:
        code_part = stock.split('_')[0] if '_' in stock else stock
        for f in os.listdir(FEATURES_DIR):
            if f.endswith(".parquet") and code_part in f:
                path = os.path.join(FEATURES_DIR, f)
                break

    if path is None or not os.path.exists(path):
        print(f"⚠️ 未找到 OHLC 文件: {stock}")
        return []

    df = pd.read_parquet(path)
    # 确保 OHLC 存在
    if 'close' in df.columns:
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df[['open', 'close']].max(axis=1)
        if 'low' not in df.columns:
            df['low'] = df[['open', 'close']].min(axis=1)
        if 'volume' not in df.columns:
            df['volume'] = 0

    # 只取最后 300 天
    # if len(df) > 300:
    #    df = df.iloc[-300:]

    records = []
    # 确定日期列
    if 'date' in df.columns:
        date_col = 'date'
    else:
        # 如果索引是 DatetimeIndex，使用索引
        if isinstance(df.index, pd.DatetimeIndex):
            date_col = None  # 将使用 idx
        else:
            # 否则尝试将索引转为字符串
            date_col = None

    for idx, row in df.iterrows():
        if date_col is not None:
            date_val = row[date_col]
        else:
            date_val = idx
        if date_val is None:
            continue
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)
        records.append({
            "date": date_str,
            "open": float(row.get('open', 0)),
            "high": float(row.get('high', 0)),
            "low": float(row.get('low', 0)),
            "close": float(row.get('close', 0)),
            "volume": float(row.get('volume', 0))
        })
    return records


def get_stock_list() -> List[str]:
    """获取所有可用股票列表（合并汇总 + 预测文件）"""
    stocks_set = set()

    # 1. 从汇总文件读取（有回测结果的股票）
    summary = load_summary()
    if isinstance(summary, list):
        for s in summary:
            name = s.get('stock')
            if name and "with_news" not in name:
                stocks_set.add(name)

    # 2. 从预测概率文件读取（补齐新增股票）
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.startswith("adaptive_probs_") and f.endswith(".csv"):
                name = f.replace("adaptive_probs_", "").replace(".csv", "")
                if name and "with_news" not in name:
                    stocks_set.add(name)

    return sorted(list(stocks_set))