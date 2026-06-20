import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

interface RollingMetricsProps {
  ohlc: any[];
  windowSize?: number;
}

const RollingMetrics: React.FC<RollingMetricsProps> = ({ ohlc, windowSize = 60 }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current || ohlc.length === 0) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current, 'dark');

    // --- 计算滚动指标 ---
    const dates = ohlc.map(d => d.date);
    const returns = ohlc.map((d, i) => {
      if (i === 0) return 0;
      return (d.close - ohlc[i - 1].close) / ohlc[i - 1].close;
    });

    const rollingSharpe: (number | null)[] = [];
    const rollingVol: (number | null)[] = [];
    const rollingReturns: (number | null)[] = [];

    for (let i = windowSize; i < returns.length; i++) {
      const slice = returns.slice(i - windowSize, i);
      const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
      const std = Math.sqrt(slice.reduce((a, b) => a + (b - avg) ** 2, 0) / (slice.length - 1 || 1));
      const sharpe = std === 0 ? 0 : (avg / std) * Math.sqrt(252);
      const totalRet = slice.reduce((a, b) => a * (1 + b), 1) - 1;
      rollingSharpe.push(sharpe);
      rollingVol.push(std * Math.sqrt(252));
      rollingReturns.push(totalRet);
    }

    // 对齐日期
    const chartDates = dates.slice(windowSize);

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(10, 14, 23, 0.92)',
        borderColor: '#1F2937',
        borderWidth: 1,
        textStyle: { color: '#E0E0E0' },
        formatter: (params: any) => {
          const idx = params[0]?.dataIndex;
          if (idx === undefined) return '';
          const date = chartDates[idx];
          const sp = rollingSharpe[idx];
          const vol = rollingVol[idx];
          const ret = rollingReturns[idx];
          return `
            <div style="font-weight:600;">${date}</div>
            <div>滚动夏普: ${sp?.toFixed(3) ?? 'N/A'}</div>
            <div>滚动波动率: ${vol ? (vol * 100).toFixed(2) : 'N/A'}%</div>
            <div>滚动收益: ${ret ? (ret * 100).toFixed(2) : 'N/A'}%</div>
          `;
        },
      },
      grid: [
        { left: '6%', right: '4%', top: '10%', height: '38%' },
        { left: '6%', right: '4%', top: '56%', height: '38%' },
      ],
      xAxis: [
        { type: 'category', data: chartDates, gridIndex: 0, axisLine: { lineStyle: { color: '#1F2937' } }, axisLabel: { color: '#6B7280', fontSize: 9 }, splitLine: { show: false } },
        { type: 'category', data: chartDates, gridIndex: 1, axisLine: { lineStyle: { color: '#1F2937' } }, axisLabel: { color: '#6B7280', fontSize: 9 }, splitLine: { show: false } },
      ],
      yAxis: [
        { scale: true, gridIndex: 0, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } }, axisLabel: { color: '#6B7280', fontSize: 10, formatter: (v: number) => v.toFixed(2) } },
        { scale: true, gridIndex: 1, splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } }, axisLabel: { color: '#6B7280', fontSize: 10, formatter: (v: number) => (v * 100).toFixed(1) + '%' } },
      ],
      series: [
        {
          name: '滚动夏普 (60日)',
          type: 'line',
          data: rollingSharpe,
          xAxisIndex: 0,
          yAxisIndex: 0,
          lineStyle: { color: '#00E5FF', width: 2 },
          areaStyle: { color: 'rgba(0, 229, 255, 0.1)' },
          symbol: 'none',
        },
        {
          name: '滚动波动率 (60日)',
          type: 'line',
          data: rollingVol.map(v => v ?? 0),
          xAxisIndex: 1,
          yAxisIndex: 1,
          lineStyle: { color: '#F59E0B', width: 2 },
          areaStyle: { color: 'rgba(245, 158, 11, 0.1)' },
          symbol: 'none',
        },
      ],
      legend: {
        data: ['滚动夏普 (60日)', '滚动波动率 (60日)'],
        textStyle: { color: '#9CA3AF', fontSize: 11 },
        top: 0,
        right: '2%',
        itemWidth: 16,
        itemHeight: 8,
      },
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, [ohlc, windowSize]);

  return <div ref={chartRef} style={{ width: '100%', height: '350px' }} />;
};

export default RollingMetrics;