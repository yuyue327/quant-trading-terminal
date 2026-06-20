import os

import numpy as np
import pandas as pd
from typing import Dict, Any, List

# 自动计算项目根目录（兼容本地Mac + Render线上，不再写死本地路径）
# 当前文件位置：ACML/quant_web_app/backend/data_loader.py
CURR_FILE_PATH = os.path.abspath(__file__)
BACKEND_FOLDER = os.path.dirname(CURR_FILE_PATH)
QUANT_WEB_FOLDER = os.path.dirname(BACKEND_FOLDER)
BASE_DIR = os.path.dirname(QUANT_WEB_FOLDER)  # 定位到ACML项目根目录

# 拼接数据目录
DATA_DIR = os.path.join(BASE_DIR, "data", "results")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# 调试打印：线上部署后可在后端日志看到真实路径，排查文件缺失
print(f"【路径调试】项目根目录 BASE_DIR = {BASE_DIR}")
print(f"【路径调试】结果目录 DATA_DIR = {DATA_DIR}")
print(f"【路径调试】特征目录 FEATURES_DIR = {FEATURES_DIR}")
print(f"【路径调试】原始行情目录 RAW_DIR = {RAW_DIR}")


def load_summary() -> Dict[str, Any]:
    """加载多股票汇总结果"""
    path = os.path.join(DATA_DIR, "adaptive_repeat_summary.csv")
    print(f"【读取文件】summary路径: {path}")
    if not os.path.exists(path):
        print(f"【警告】汇总文件不存在: {path}")
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
    if path is None and os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            if f.startswith("adaptive_probs_") and stock in f:
                path = os.path.join(DATA_DIR, f)
                break
    if path is None or not os.path.exists(path):
        print(f"【警告】未找到{stock}预测概率文件")
        return []
    print(f"【读取文件】{stock}概率文件路径: {path}")
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
    # 优先读取raw下csv（Render线上仅上传raw轻量化csv，无parquet）
    raw_path = None
    parquet_path = None

    # 先匹配raw原始csv
    raw_names = [
        f"{stock}.csv",
        f"{stock.replace('_', '.')}.csv",
    ]
    for name in raw_names:
        test_path = os.path.join(RAW_DIR, name)
        if os.path.exists(test_path):
            raw_path = test_path
            break

    # 再尝试匹配features下parquet（本地完整数据使用）
    possible_names = [
        f"{stock}.parquet",
        f"{stock.replace('_', '.')}.parquet",
    ]
    for name in possible_names:
        test_path = os.path.join(FEATURES_DIR, name)
        if os.path.exists(test_path):
            parquet_path = test_path
            break

    # 模糊匹配兜底
    if raw_path is None and os.path.exists(RAW_DIR):
        code_part = stock.split('_')[0] if '_' in stock else stock
        for f in os.listdir(RAW_DIR):
            if f.endswith(".csv") and code_part in f:
                raw_path = os.path.join(RAW_DIR, f)
                break
    if parquet_path is None and os.path.exists(FEATURES_DIR):
        code_part = stock.split('_')[0] if '_' in stock else stock
        for f in os.listdir(FEATURES_DIR):
            if f.endswith(".parquet") and code_part in f:
                parquet_path = os.path.join(FEATURES_DIR, f)
                break

    # 优先使用csv，无csv再用parquet
    path = raw_path if raw_path is not None else parquet_path
    if path is None or not os.path.exists(path):
        print(f"⚠️ 未找到 OHLC 文件: {stock}")
        return []
    print(f"【读取文件】{stock} K线数据源: {path}")

    # 区分csv/parquet读取
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    # 确保 OHLC 字段齐全
    if 'close' in df.columns:
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df[['open', 'close']].max(axis=1)
        if 'low' not in df.columns:
            df['low'] = df[['open', 'close']].min(axis=1)
        if 'volume' not in df.columns:
            df['volume'] = 0

    records = []
    # 确定日期列
    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    else:
        if isinstance(df.index, pd.DatetimeIndex):
            date_col = None
        else:
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
    """获取所有可用股票列表（合并汇总 + 预测文件 + raw原始csv兜底）"""
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

    # 3. 兜底：从raw原始csv提取标的（Render线上无results/features时也能返回列表）
    if os.path.exists(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            if f.endswith(".csv") and "with_news" not in f:
                stock_name = f.replace(".csv", "")
                stocks_set.add(stock_name)

    stock_list = sorted(list(stocks_set))
    print(f"【股票列表】共加载 {len(stock_list)} 只标的: {stock_list}")
    return stock_list