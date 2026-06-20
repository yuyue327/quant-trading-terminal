import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { useApp } from '../context/AppContext';

type SubChartType = 'macd' | 'rsi' | 'volume';
type TimeFrame = 'daily' | 'weekly' | 'monthly';

const CandlestickChart: React.FC = () => {
  const { ohlc, selected } = useApp();
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [subChart, setSubChart] = useState<SubChartType>('macd');
  const [timeframe, setTimeframe] = useState<TimeFrame>('daily');

  useEffect(() => {
    if (!chartRef.current || ohlc.length === 0) {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
      return;
    }

    if (chartInstance.current) {
      chartInstance.current.dispose();
      chartInstance.current = null;
    }

    chartInstance.current = echarts.init(chartRef.current, 'dark');

    // --- 数据聚合（周期切换） ---
    const getAggregatedData = (data: any[], tf: TimeFrame) => {
      if (tf === 'daily') return data;
      const groupSize = tf === 'weekly' ? 5 : 20;
      const result = [];
      for (let i = 0; i < data.length; i += groupSize) {
        const chunk = data.slice(i, i + groupSize);
        if (chunk.length === 0) continue;
        result.push({
          date: chunk[0].date,
          open: chunk[0].open,
          close: chunk[chunk.length - 1].close,
          high: Math.max(...chunk.map(d => d.high)),
          low: Math.min(...chunk.map(d => d.low)),
          volume: chunk.reduce((s, d) => s + d.volume, 0),
        });
      }
      return result;
    };

    const ohlcData = getAggregatedData(ohlc, timeframe);
    const dates = ohlcData.map(d => d.date);
    const values = ohlcData.map(d => [d.open, d.close, d.low, d.high]);
    const volumes = ohlcData.map(d => Math.max(d.volume || 0, 0));
    const closePrices = ohlcData.map(d => d.close);

    // --- 成交量颜色 ---
    const volumeColors = ohlcData.map(d =>
      d.close >= d.open ? 'rgba(0, 245, 160, 0.6)' : 'rgba(255, 0, 80, 0.6)'
    );

    // --- 技术指标（MACD, RSI）---
    function calculateEMA(data: number[], period: number): number[] {
      const result: number[] = [];
      const multiplier = 2 / (period + 1);
      let ema = data[0] || 0;
      for (let i = 0; i < data.length; i++) {
        if (i === 0) ema = data[i];
        else ema = (data[i] - ema) * multiplier + ema;
        result.push(ema);
      }
      return result;
    }

    const ema12 = calculateEMA(closePrices, 12);
    const ema26 = calculateEMA(closePrices, 26);
    const macdLine = ema12.map((v, i) => v - ema26[i]);
    const signalLine = calculateEMA(macdLine, 9);
    const histogram = macdLine.map((v, i) => v - signalLine[i]);

    function calculateRSI(data: number[], period: number): number[] {
      const result: number[] = [];
      let gain = 0, loss = 0;
      for (let i = 1; i < data.length; i++) {
        const diff = data[i] - data[i - 1];
        if (diff >= 0) gain += diff;
        else loss -= diff;
        if (i >= period) {
          const avgGain = gain / period;
          const avgLoss = loss / period;
          const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
          result.push(100 - 100 / (1 + rs));
          const oldDiff = data[i - period + 1] - data[i - period];
          if (oldDiff >= 0) gain -= oldDiff;
          else loss += oldDiff;
        } else {
          result.push(50);
        }
      }
      return result;
    }
    const rsiValues = calculateRSI(closePrices, 14);

    // --- 副图配置 ---
    let subChartSeries: any[] = [];
    let subChartYAxis: any = {};

    switch (subChart) {
      case 'macd':
        subChartSeries = [
          {
            name: 'MACD',
            type: 'line',
            data: macdLine,
            xAxisIndex: 1,
            yAxisIndex: 1,
            lineStyle: { color: '#00E5FF', width: 1.5 },
            symbol: 'none',
          },
          {
            name: '信号线',
            type: 'line',
            data: signalLine,
            xAxisIndex: 1,
            yAxisIndex: 1,
            lineStyle: { color: '#F59E0B', width: 1.5 },
            symbol: 'none',
          },
          {
            name: 'MACD柱',
            type: 'bar',
            data: histogram,
            xAxisIndex: 1,
            yAxisIndex: 1,
            itemStyle: {
              color: (params: any) =>
                params.value >= 0 ? 'rgba(0,245,160,0.6)' : 'rgba(255,0,80,0.6)',
            },
          },
        ];
        subChartYAxis = {
          scale: true,
          gridIndex: 1,
          splitLine: { show: false },
          axisLabel: { color: '#6B7280', fontSize: 10 },
        };
        break;

      case 'rsi':
        subChartSeries = [
          {
            name: 'RSI',
            type: 'line',
            data: rsiValues,
            xAxisIndex: 1,
            yAxisIndex: 1,
            smooth: true,
            lineStyle: { color: '#8B5CF6', width: 2 },
            areaStyle: { color: 'rgba(139,92,246,0.15)' },
            symbol: 'none',
            markLine: {
              silent: true,
              data: [
                {
                  yAxis: 70,
                  lineStyle: { color: '#FF0050', type: 'dashed' },
                  label: { color: '#FF0050', formatter: '超买 70', fontSize: 10 },
                },
                {
                  yAxis: 30,
                  lineStyle: { color: '#00F5A0', type: 'dashed' },
                  label: { color: '#00F5A0', formatter: '超卖 30', fontSize: 10 },
                },
              ],
            },
          },
        ];
        subChartYAxis = {
          scale: true,
          gridIndex: 1,
          splitLine: { show: false },
          axisLabel: { color: '#6B7280', fontSize: 10 },
          min: 0,
          max: 100,
        };
        break;

      case 'volume':
      default:
        subChartSeries = [
          {
            name: '成交量',
            type: 'bar',
            data: volumes,
            xAxisIndex: 1,
            yAxisIndex: 1,
            barWidth: '60%',
            itemStyle: {
              color: (params: any) =>
                volumeColors[params.dataIndex] || 'rgba(100,100,100,0.5)',
            },
          },
        ];
        subChartYAxis = {
          scale: true,
          gridIndex: 1,
          splitLine: { show: false },
          axisLabel: {
            color: '#6B7280',
            fontSize: 10,
            formatter: (v: number) => {
              if (v >= 1e8) return (v / 1e8).toFixed(1) + '亿';
              if (v >= 1e4) return (v / 1e4).toFixed(0) + '万';
              return v.toFixed(0);
            },
          },
        };
        break;
    }

    // --- ECharts 配置 ---
    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      title: {
        text: `📊 ${selected} | 自适应量化系统${timeframe !== 'daily' ? ` (${timeframe === 'weekly' ? '周线' : '月线'})` : ''}`,
        left: '2%',
        top: 0,
        textStyle: {
          color: 'rgba(255,255,255,0.15)',
          fontSize: 14,
          fontWeight: 300,
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        backgroundColor: 'rgba(10,14,23,0.92)',
        borderColor: '#1F2937',
        borderWidth: 1,
        textStyle: { color: '#E0E0E0', fontSize: 12 },
        formatter: (params: any) => {
          const dataIndex = params[0]?.dataIndex;
          if (dataIndex === undefined) return '';
          const d = ohlcData[dataIndex];
          return `
            <div style="font-weight:600;margin-bottom:6px;">${d.date}</div>
            <div>📈 开盘: ${d.open.toFixed(2)}</div>
            <div>📉 收盘: ${d.close.toFixed(2)}</div>
            <div>📊 最高: ${d.high.toFixed(2)} | 最低: ${d.low.toFixed(2)}</div>
            <div style="margin-top:6px;border-top:1px solid #1F2937;padding-top:6px;">
              📊 成交量: ${d.volume?.toFixed(0) || 'N/A'}
            </div>
          `;
        },
      },
      grid: [
        { left: '6%', right: '4%', top: '10%', height: '58%' },
        { left: '6%', right: '4%', top: '74%', height: '20%' },
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          gridIndex: 0,
          axisLine: { lineStyle: { color: '#1F2937' } },
          axisLabel: { color: '#6B7280', fontSize: 10 },
          splitLine: { show: false },
        },
        {
          type: 'category',
          data: dates,
          gridIndex: 1,
          axisLine: { lineStyle: { color: '#1F2937' } },
          axisLabel: { color: '#6B7280', fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      yAxis: [
        {
          scale: true,
          gridIndex: 0,
          splitLine: {
            lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' },
          },
          axisLabel: {
            color: '#6B7280',
            fontSize: 10,
            formatter: (v: number) => v.toFixed(2),
          },
        },
        subChartYAxis,
      ],
      series: [
        // --- K线 ---
        {
          name: 'K线',
          type: 'candlestick',
          data: values,
          xAxisIndex: 0,
          yAxisIndex: 0,
          itemStyle: {
            color: '#00F5A0',
            color0: '#FF0050',
            borderColor: '#00F5A0',
            borderColor0: '#FF0050',
            borderWidth: 1,
          },
        },
        // --- MA5 ---
        {
          name: 'MA5',
          type: 'line',
          data: closePrices.map((_, i, arr) => {
            if (i < 4) return null;
            const sum = arr.slice(i - 4, i + 1).reduce((a, b) => a + b, 0);
            return +(sum / 5).toFixed(2);
          }),
          xAxisIndex: 0,
          yAxisIndex: 0,
          lineStyle: { color: '#F59E0B', width: 1 },
          symbol: 'none',
          connectNulls: true,
        },
        // --- MA20 ---
        {
          name: 'MA20',
          type: 'line',
          data: closePrices.map((_, i, arr) => {
            if (i < 19) return null;
            const sum = arr.slice(i - 19, i + 1).reduce((a, b) => a + b, 0);
            return +(sum / 20).toFixed(2);
          }),
          xAxisIndex: 0,
          yAxisIndex: 0,
          lineStyle: { color: '#00E5FF', width: 1 },
          symbol: 'none',
          connectNulls: true,
        },
        // --- 副图 ---
        ...subChartSeries,
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: 0,
          end: 100,
        },
        {
          type: 'slider',
          xAxisIndex: [0, 1],
          start: 0,
          end: 100,
          height: 20,
          bottom: 5,
          borderColor: '#1F2937',
          backgroundColor: 'rgba(17,24,39,0.6)',
          fillerColor: 'rgba(0,229,255,0.15)',
          handleStyle: { color: '#00E5FF' },
          textStyle: { color: '#6B7280', fontSize: 10 },
        },
      ],
      legend: {
        data: ['K线', 'MA5', 'MA20', ...subChartSeries.map(s => s.name)],
        textStyle: { color: '#9CA3AF', fontSize: 11 },
        top: 2,
        right: '2%',
        itemWidth: 16,
        itemHeight: 8,
      },
      animation: true,
      animationDuration: 500,
      animationEasing: 'cubicOut',
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, [ohlc, selected, subChart, timeframe]);

  // --- 副图切换按钮 ---
  const subChartOptions: { value: SubChartType; label: string }[] = [
    { value: 'macd', label: '📈 MACD' },
    { value: 'rsi', label: '📉 RSI' },
    { value: 'volume', label: '📊 成交量' },
  ];

  return (
    <div>
      <div style={{ display: 'flex', gap: '8px', marginBottom: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
        {subChartOptions.map(opt => (
          <button
            key={opt.value}
            onClick={() => setSubChart(opt.value)}
            style={{
              background: subChart === opt.value ? '#00E5FF' : 'rgba(255,255,255,0.05)',
              color: subChart === opt.value ? '#0A0E17' : '#9CA3AF',
              border: '1px solid ' + (subChart === opt.value ? '#00E5FF' : '#1F2937'),
              borderRadius: '6px',
              padding: '4px 12px',
              fontSize: '12px',
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontWeight: subChart === opt.value ? 600 : 400,
            }}
          >
            {opt.label}
          </button>
        ))}
        <div style={{ width: '1px', height: '24px', background: '#1F2937', margin: '0 4px' }} />
        <span style={{ color: '#6B7280', fontSize: '11px' }}>周期:</span>
        {(['daily', 'weekly', 'monthly'] as TimeFrame[]).map(t => (
          <button
            key={t}
            onClick={() => setTimeframe(t)}
            style={{
              background: timeframe === t ? '#00E5FF' : 'rgba(255,255,255,0.05)',
              color: timeframe === t ? '#0A0E17' : '#9CA3AF',
              border: '1px solid ' + (timeframe === t ? '#00E5FF' : '#1F2937'),
              borderRadius: '4px',
              padding: '2px 10px',
              fontSize: '10px',
              cursor: 'pointer',
            }}
          >
            {t === 'daily' ? '日' : t === 'weekly' ? '周' : '月'}
          </button>
        ))}
      </div>
      <div ref={chartRef} style={{ width: '100%', height: '650px', minHeight: '400px' }} />
    </div>
  );
};

export default CandlestickChart;