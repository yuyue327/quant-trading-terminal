#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step34_fetch_news_sentiment.py
获取股票新闻并使用 LLM (DeepSeek 或 FinBERT) 进行情感打分，替换模拟的 llm_score
"""

import os
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

# ==================== 配置 ====================
STOCK_CODE = "600036"  # 招商银行
STOCK_NAME = "A_sh.600036_招商银行"
FEATURE_PATH = f"data/features/{STOCK_NAME}.parquet"
OUTPUT_PATH = f"data/features/{STOCK_NAME}_with_news.parquet"

# 新闻获取参数
NEWS_DAYS_BACK = 365 * 3  # 获取近3年的新闻（可根据数据时间范围调整）
USE_DEEPSEEK = True  # 设为 False 则使用 FinBERT

# DeepSeek API
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if USE_DEEPSEEK and not DEEPSEEK_API_KEY:
    print("警告: 未设置 DEEPSEEK_API_KEY，将回退到 FinBERT")
    USE_DEEPSEEK = False


# ==================== 1. 获取新闻 ====================
def fetch_news_akshare(stock_code, days_back=365 * 3):
    """使用 akshare 获取股票新闻标题（新浪财经）"""
    try:
        import akshare as ak
        # 获取新闻列表（部分接口可能需要指定日期范围）
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
        # 新浪财经新闻接口
        news_df = ak.stock_news_em(symbol=stock_code, start_date=start_date, end_date=end_date)
        if news_df is None or news_df.empty:
            print("未获取到新闻，尝试备用接口...")
            # 备用：和讯网新闻
            news_df = ak.stock_news_baidu(symbol=stock_code)
        if news_df is None or news_df.empty:
            raise ValueError("无法获取新闻数据，请检查网络或 akshare 接口")
        # 提取标题和日期
        news = news_df[['title', 'date']].copy()
        news['date'] = pd.to_datetime(news['date']).dt.date
        print(f"获取到 {len(news)} 条新闻")
        return news
    except Exception as e:
        print(f"获取新闻失败: {e}")
        # 返回模拟数据（避免中断流程）
        print("生成模拟新闻用于测试...")
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        titles = [f"模拟新闻 {i}" for i in range(100)]
        return pd.DataFrame({'date': dates.date, 'title': titles})


# ==================== 2. 情感打分（DeepSeek） ====================
def call_deepseek_sentiment(text, retries=3):
    """调用 DeepSeek API 对单条新闻打分，返回 0-100 整数"""
    import requests
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""你是一个金融情感分析专家。请分析以下新闻对股票（招商银行）的短期影响，给出一个0到100之间的整数分数，0表示极度负面（股价可能下跌），100表示极度正面（股价可能上涨）。只输出数字，不要有其他内容。

新闻：{text}

分数："""
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 10
    }
    for attempt in range(retries):
        try:
            resp = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload,
                                 timeout=30)
            if resp.status_code == 200:
                raw = resp.json()['choices'][0]['message']['content'].strip()
                match = re.search(r'\b([0-9]|[1-9][0-9]|100)\b', raw)
                if match:
                    return int(match.group())
                digits = ''.join([c for c in raw if c.isdigit()])
                if digits:
                    return min(100, int(digits[:3]))
                return 50
            else:
                time.sleep(2 ** attempt)
        except:
            time.sleep(2 ** attempt)
    return 50


# ==================== 3. 情感打分（FinBERT 本地） ====================
def load_finbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def call_finbert_sentiment(text, tokenizer, model):
    """使用 FinBERT 返回 0-100 的分数（正面概率 * 100）"""
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT 输出: [负面, 中性, 正面]
        pos_prob = probs[0, 2].item()
        return int(pos_prob * 100)


# ==================== 4. 主流程 ====================
def main():
    print("=" * 60)
    print("step34：获取真实新闻并生成 LLM 情感分数")
    print("=" * 60)

    # 获取新闻
    print(f"获取股票 {STOCK_CODE} 近期的新闻...")
    news_df = fetch_news_akshare(STOCK_CODE, NEWS_DAYS_BACK)
    if news_df.empty:
        print("未获取到任何新闻，请检查网络或修改日期范围")
        return

    # 情感打分
    print("开始情感打分...")
    if USE_DEEPSEEK:
        print("使用 DeepSeek API 进行打分（可能需要较长时间）")
        scores = []
        for title in tqdm(news_df['title'], desc="DeepSeek打分"):
            score = call_deepseek_sentiment(title)
            scores.append(score)
            time.sleep(0.2)  # 避免 API 速率限制
    else:
        print("使用 FinBERT 本地模型打分（速度较快）")
        tokenizer, model = load_finbert()
        scores = []
        for title in tqdm(news_df['title'], desc="FinBERT打分"):
            score = call_finbert_sentiment(title, tokenizer, model)
            scores.append(score)

    news_df['sentiment'] = scores

    # 按日期聚合每日平均情感分
    daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    print(f"聚合后得到 {len(daily_sentiment)} 个有新闻的日期")

    # 加载原始特征数据
    df_feat = pd.read_parquet(FEATURE_PATH)
    if 'date' in df_feat.columns:
        df_feat['date'] = pd.to_datetime(df_feat['date'])
    else:
        df_feat['date'] = df_feat.index  # 假设索引是日期
        df_feat.reset_index(drop=False, inplace=True)

    # 合并情感分数：左连接，缺失值用 50（中性）填充
    df_merged = df_feat.merge(daily_sentiment, on='date', how='left')
    df_merged['llm_score'] = df_merged['sentiment'].fillna(50).astype(int)
    df_merged.drop(columns=['sentiment'], inplace=True)

    # 确保 llm_score 在 0-100 范围
    df_merged['llm_score'] = df_merged['llm_score'].clip(0, 100)

    # 保存新特征文件
    df_merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"更新后的特征文件已保存至: {OUTPUT_PATH}")
    print(
        f"llm_score 统计: min={df_merged['llm_score'].min()}, mean={df_merged['llm_score'].mean():.1f}, max={df_merged['llm_score'].max()}")

    # 可选：绘制情感时间序列
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.figure(figsize=(12, 4))
    plt.plot(df_merged['date'], df_merged['llm_score'], alpha=0.7)
    plt.title('新闻情感得分（每日平均）')
    plt.xlabel('日期')
    plt.ylabel('情感分数 (0=负面, 100=正面)')
    plt.tight_layout()
    os.makedirs("data/figures", exist_ok=True)
    plt.savefig("data/figures/news_sentiment_timeseries.png", dpi=150)
    plt.close()
    print("情感时间序列图已保存至 data/figures/news_sentiment_timeseries.png")

    print("step34 完成。")


if __name__ == "__main__":
    main()