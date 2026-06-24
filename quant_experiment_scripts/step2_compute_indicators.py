#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版：处理A股和美股（美股有多行表头）
"""

import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_DATA_DIR = "data/raw"
FEATURE_DIR = "data/features"
os.makedirs(FEATURE_DIR, exist_ok=True)

TECH_INDICATORS = [
    'MA5', 'MA10', 'MA20', 'MA60',
    'EMA12', 'EMA26',
    'MACD', 'MACD_signal', 'MACD_hist',
    'RSI',
    'BB_upper', 'BB_middle', 'BB_lower',
    'BB_width', 'BB_pct',
    'ATR',
    'volume_ratio',
    'pct_change',
    'high_low_pct',
    'close_position',
    'volatility_5', 'volatility_20',
]


def read_stock_csv(file_path):
    """智能读取CSV，处理A股和美股的不同格式"""
    # 先读取前几行判断格式
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()

    # 如果第一行是列名且包含多个逗号分隔，第二行以逗号开头（如",MSFT,..."），则是双表头
    if second_line.startswith(',') or second_line.count(',') == first_line.count(','):
        # 美股格式：跳过前两行，列名从第一行取（去掉第一个空列）
        columns = first_line.split(',')
        # 去掉可能的空列名（如第一个）
        columns = [c for c in columns if c != '']
        # 读取数据，跳过前两行
        df = pd.read_csv(file_path, skiprows=2, names=columns, header=None)
    else:
        # A股格式：正常读取
        df = pd.read_csv(file_path)

    # 统一列名：将首字母大写的改为小写
    rename_map = {}
    for col in df.columns:
        if col.lower() == 'date':
            rename_map[col] = 'date'
        elif col.lower() == 'open':
            rename_map[col] = 'open'
        elif col.lower() == 'high':
            rename_map[col] = 'high'
        elif col.lower() == 'low':
            rename_map[col] = 'low'
        elif col.lower() == 'close':
            rename_map[col] = 'close'
        elif col.lower() == 'volume':
            rename_map[col] = 'volume'
        elif col.lower() == 'adj close':
            rename_map[col] = 'adj_close'
    df = df.rename(columns=rename_map)

    # 确保有close列，如果没有则尝试用adj_close
    if 'close' not in df.columns:
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
        else:
            raise ValueError(f"Missing 'close' or 'Adj Close' in {file_path}")

    # 确保所有必需的列存在
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}")

    # 转换日期
    df['date'] = pd.to_datetime(df['date'])
    # 数值转换
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 删除无效行
    df = df.dropna(subset=required)
    df = df.sort_values('date').reset_index(drop=True)

    return df


def add_technical_indicators(df):
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()

    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-9)

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']

    df['pct_change'] = df['close'].pct_change() * 100
    df['high_low_pct'] = (df['high'] - df['low']) / df['low'] * 100
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

    df['volatility_5'] = df['pct_change'].rolling(5).std()
    df['volatility_20'] = df['pct_change'].rolling(20).std()

    return df


def add_labels(df, horizon=1):
    df = df.copy()
    future_close = df['close'].shift(-horizon)
    df['label'] = (future_close > df['close']).astype(int)
    return df


def process_stock(file_path):
    stock_name = os.path.basename(file_path).replace('.csv', '')
    print(f"Processing {stock_name}...")
    df = read_stock_csv(file_path)
    df = add_technical_indicators(df)
    df = add_labels(df)

    feature_cols = ['date', 'close', 'volume'] + TECH_INDICATORS + ['label']
    existing_cols = [c for c in feature_cols if c in df.columns]
    df_features = df[existing_cols].copy()
    df_features = df_features.iloc[60:].reset_index(drop=True)

    output_path = os.path.join(FEATURE_DIR, f"{stock_name}.parquet")
    df_features.to_parquet(output_path, index=False)
    print(f"  Saved {len(df_features)} rows to {output_path}")
    return df_features


def main():
    stock_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    print(f"Found {len(stock_files)} stock files")

    all_features = []
    for file_path in tqdm(stock_files, desc="Processing"):
        try:
            feat = process_stock(file_path)
            all_features.append(feat)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if all_features:
        combined = pd.concat(all_features, keys=[os.path.basename(f).replace('.csv', '') for f in stock_files],
                             names=['stock', 'idx']).reset_index(level='stock').reset_index(drop=True)
        combined.to_parquet(os.path.join(FEATURE_DIR, "all_stocks_features.parquet"), index=False)
        print("Saved combined features.")

    print("Done.")


if __name__ == "__main__":
    main()