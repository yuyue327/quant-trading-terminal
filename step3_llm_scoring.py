#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用DeepSeek API为每个交易日生成LLM打分（0-100）
API Key从环境变量 DEEPSEEK_API_KEY 读取
"""

import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

# ========== 配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请设置环境变量 DEEPSEEK_API_KEY，例如: export DEEPSEEK_API_KEY='sk-xxx'")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-chat"  # 或 "deepseek-reasoner"

FEATURE_DIR = "data/features"
LLM_SCORE_DIR = "data/llm_scores"
os.makedirs(LLM_SCORE_DIR, exist_ok=True)

PROMPT_COLS = [
    'close', 'volume', 'pct_change', 'high_low_pct',
    'MA5', 'MA20', 'RSI', 'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower', 'BB_pct', 'volume_ratio', 'ATR'
]


def build_prompt(df_window):
    recent = df_window.tail(20).copy()
    lines = []
    for idx, row in recent.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        values = []
        for col in PROMPT_COLS:
            val = row[col]
            if pd.notna(val):
                values.append(f"{val:.2f}")
            else:
                values.append("NaN")
        lines.append(f"{date_str}: " + ", ".join(values))

    prompt = f"""你是一个金融分析师。基于以下最近20个交易日的技术指标（日期, close, volume, pct_change, amplitude, MA5, MA20, RSI, MACD, MACD_signal, BB_upper, BB_lower, BB_pct, volume_ratio, ATR），预测下一个交易日收盘价上涨的可能性。请只输出一个0到100之间的整数分数，分数越高表示上涨可能性越大。不要输出任何其他内容。

数据：
{chr(10).join(lines)}

你的预测分数（0-100整数）："""
    return prompt


def call_llm(prompt, retries=3):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 10
    }
    for attempt in range(retries):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                raw = result['choices'][0]['message']['content'].strip()
                match = re.search(r'\b([0-9]|[1-9][0-9]|100)\b', raw)
                if match:
                    return int(match.group())
                digits = ''.join([c for c in raw if c.isdigit()])
                if digits:
                    return min(100, int(digits[:3]))
                return 50
            else:
                print(f"API error {response.status_code}: {response.text}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"Request failed: {e}")
            time.sleep(2 ** attempt)
    return 50


def process_stock(stock_name, limit=None):
    feat_path = os.path.join(FEATURE_DIR, f"{stock_name}.parquet")
    if not os.path.exists(feat_path):
        print(f"Feature file not found: {feat_path}")
        return None

    df = pd.read_parquet(feat_path)
    if 'date' not in df.columns:
        print(f"No date column in {stock_name}")
        return None

    out_path = os.path.join(LLM_SCORE_DIR, f"{stock_name}.parquet")
    if os.path.exists(out_path):
        print(f"Skipping {stock_name}, already exists")
        return pd.read_parquet(out_path)

    scores = []
    total = len(df) - 20
    if limit:
        total = min(total, limit)

    for i in tqdm(range(20, 20 + total), desc=stock_name, leave=False):
        window = df.iloc[:i + 1]
        prompt = build_prompt(window)
        score = call_llm(prompt)
        scores.append({
            'date': df.iloc[i]['date'],
            'llm_score': score
        })
        time.sleep(0.2)  # 避免请求过快

    scores_df = pd.DataFrame(scores)
    scores_df.to_parquet(out_path, index=False)
    print(f"Saved {len(scores_df)} scores for {stock_name}")
    return scores_df


def main():
    stock_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.parquet') and f != 'all_stocks_features.parquet']
    stock_names = [f.replace('.parquet', '') for f in stock_files]
    print(f"Found {len(stock_names)} stocks")

    # 可选：测试时只处理前1只股票
    # stock_names = stock_names[:1]

    for name in tqdm(stock_names, desc="Overall"):
        process_stock(name)

    # 合并所有分数
    all_scores = []
    for name in stock_names:
        path = os.path.join(LLM_SCORE_DIR, f"{name}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['stock'] = name
            all_scores.append(df)
    if all_scores:
        combined = pd.concat(all_scores, ignore_index=True)
        combined.to_parquet(os.path.join(LLM_SCORE_DIR, "all_scores.parquet"), index=False)
        print("Saved combined scores.")


if __name__ == "__main__":
    main()