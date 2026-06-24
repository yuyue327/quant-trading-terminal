#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅下载A股数据（使用baostock，国内网络可用）
"""

import os
import time
import pandas as pd
import baostock as bs

START_DATE = "2021-01-01"
END_DATE = "2026-06-01"

# A股列表 (baostock代码格式: sh.600036 或 sz.000001)
A_STOCKS = [
    ("sh.600036", "招商银行"),
    ("sz.000001", "平安银行"),
    ("sz.002142", "宁波银行"),
    ("sh.600030", "中信证券"),
    ("sh.601688", "华泰证券"),
    ("sz.300059", "东方财富"),
    ("sh.600519", "贵州茅台"),
    ("sz.000858", "五粮液"),
    ("sz.000568", "泸州老窖"),
    ("sz.000333", "美的集团"),
    ("sz.000651", "格力电器"),
    ("sh.600887", "伊利股份"),
    ("sz.300750", "宁德时代"),
    ("sz.002594", "比亚迪"),
    ("sh.601012", "隆基绿能"),
]

RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def download_a_stock(code, name):
    print(f"正在下载 A股: {name} ({code}) ...")
    try:
        bs.login()
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume",
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="d",
            adjustflag="2"  # 前复权
        )
        if rs.error_code != '0':
            print(f"  错误: {name} 查询失败 - {rs.error_msg}")
            bs.logout()
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        if not data_list:
            print(f"  警告: {name} 无数据")
            bs.logout()
            return None

        df = pd.DataFrame(data_list, columns=["date", "open", "high", "low", "close", "volume"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        filename = os.path.join(RAW_DATA_DIR, f"A_{code}_{name}.csv")
        df.to_csv(filename, index=False)
        print(f"  成功: {name}, {len(df)} 行, 保存至 {filename}")
        bs.logout()
        time.sleep(0.3)
        return df
    except Exception as e:
        print(f"  错误: {name} 下载失败 - {e}")
        bs.logout()
        return None


def main():
    print("=" * 60)
    print("开始下载A股数据 (baostock)")
    print(f"时间范围: {START_DATE} 至 {END_DATE}")
    print("=" * 60)
    success = 0
    for code, name in A_STOCKS:
        if download_a_stock(code, name):
            success += 1
    print(f"\n下载完成: 成功 {success}/{len(A_STOCKS)}")
    print("数据保存在:", RAW_DATA_DIR)


if __name__ == "__main__":
    main()