import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

interface OrderBookDepthProps {
  currentPrice: number;
  stockName: string;
}

const OrderBookDepth: React.FC<OrderBookDepthProps> = ({ currentPrice, stockName }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current, 'dark');

    // 生成模拟订单簿数据（围绕当前价格）
    const bids: { price: number; volume: number }[] = [];
    const asks: { price: number; volume: number }[] = [];

    for (let i = 0; i < 10; i++) {
      const spread = 0.002 * Math.random() + 0.005;
      bids.push({
        price: currentPrice * (1 - (i + 1) * spread * 0.5),
        volume: Math.floor(Math.random() * 8000 + 2000),
      });
      asks.push({
        price: currentPrice * (1 + (i + 1) * spread * 0.5),
        volume: Math.floor(Math.random() * 8000 + 2000),
      });
    }

    // 合并数据
    const bidPrices = bids.map(b => b.price);
    const bidVolumes = bids.map(b => -b.volume); // 负数表示买盘（左侧）
    const askPrices = asks.map(a => a.price);
    const askVolumes = asks.map(a => a.volume);

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(10,14,23,0.92)',
        borderColor: '#1F2937',
        borderWidth: 1,
        textStyle: { color: '#E0E0E0' },
        formatter: (params: any) => {
          const p = params[0];
          return `${p.axisValue}<br>数量: ${Math.abs(p.value).toFixed(0)}`;
        },
      },
      grid: {
        left: '8%',
        right: '8%',
        top: '8%',
        bottom: '8%',
      },
      xAxis: {
        type: 'value',
        splitLine: { show: false },
        axisLabel: { color: '#6B7280', fontSize: 10 },
        axisLine: { lineStyle: { color: '#1F2937' } },
      },
      yAxis: {
        type: 'category',
        data: [
          ...bidPrices.map(p => p.toFixed(2)),
          ...askPrices.map(p => p.toFixed(2)),
        ],
        splitLine: { show: false },
        axisLabel: { color: '#6B7280', fontSize: 9 },
        axisLine: { lineStyle: { color: '#1F2937' } },
      },
      series: [
        {
          name: '买盘',
          type: 'bar',
          data: [
            ...bidVolumes.map(v => ({ value: v, itemStyle: { color: 'rgba(0,245,160,0.8)' } })),
            ...askVolumes.map(() => ({ value: 0, itemStyle: { color: 'transparent' } })),
          ],
          barWidth: '70%',
          barGap: 0,
          label: {
            show: true,
            position: 'left',
            formatter: (p: any) => {
              if (p.value < 0) return Math.abs(p.value).toFixed(0);
              return '';
            },
            color: '#00F5A0',
            fontSize: 9,
          },
        },
        {
          name: '卖盘',
          type: 'bar',
          data: [
            ...bidVolumes.map(() => ({ value: 0, itemStyle: { color: 'transparent' } })),
            ...askVolumes.map(v => ({ value: v, itemStyle: { color: 'rgba(255,0,80,0.8)' } })),
          ],
          barWidth: '70%',
          barGap: 0,
          label: {
            show: true,
            position: 'right',
            formatter: (p: any) => {
              if (p.value > 0) return p.value.toFixed(0);
              return '';
            },
            color: '#FF0050',
            fontSize: 9,
          },
        },
        {
          name: '中间价',
          type: 'scatter',
          data: [
            {
              value: [0, currentPrice],
              symbol: 'diamond',
              symbolSize: 16,
              itemStyle: { color: '#00E5FF' },
              label: { show: true, formatter: `¥${currentPrice.toFixed(2)}`, color: '#00E5FF', fontSize: 11, fontWeight: 600 },
            },
          ],
          xAxisIndex: 0,
          yAxisIndex: 0,
        },
      ],
      legend: {
        data: ['买盘', '卖盘', '中间价'],
        textStyle: { color: '#9CA3AF', fontSize: 10 },
        top: 0,
        right: '2%',
        itemWidth: 12,
        itemHeight: 6,
      },
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, [currentPrice]);

  return <div ref={chartRef} style={{ width: '100%', height: '220px' }} />;
};

export default OrderBookDepth;