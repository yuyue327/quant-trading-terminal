import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { useApp } from '../context/AppContext';

interface PerformanceDashboardProps {
  ohlc: any[];
  probs: any[];
  stockName: string;
}

const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({ ohlc, probs, stockName }) => {
  const { theme } = useApp();
  const isDark = theme === 'dark';

  const equityChartRef = useRef<HTMLDivElement>(null);
  const drawdownChartRef = useRef<HTMLDivElement>(null);
  const equityInstance = useRef<echarts.ECharts | null>(null);
  const drawdownInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ohlc.length) return;

    // ===== 资金曲线 =====
    if (equityChartRef.current) {
      if (equityInstance.current) equityInstance.current.dispose();
      equityInstance.current = echarts.init(equityChartRef.current, isDark ? 'dark' : 'light');

      const dates = ohlc.map(d => d.date);
      const closes = ohlc.map(d => d.close);
      const norm = closes.map((v, i) => (v / closes[0]) * 100);

      // 策略净值（基于概率信号，如果有的话）
      let strategyNav: number[];
      if (probs && probs.length > 0) {
        strategyNav = [100];
        for (let i = 1; i < closes.length; i++) {
          const ret = (closes[i] - closes[i - 1]) / closes[i - 1];
          const prob = i < probs.length ? (probs[i]?.prob ?? 0.5) : 0.5;
          const signal = prob > 0.55 ? 1 : prob < 0.45 ? -1 : 0;
          strategyNav.push(strategyNav[i - 1] * (1 + ret * signal * 1.2));
        }
      } else {
        // 没有概率数据时使用价格归一化
        strategyNav = norm.map(v => v);
      }

      equityInstance.current.setOption({
        backgroundColor: 'transparent',
        tooltip: {
          trigger: 'axis',
          backgroundColor: isDark ? 'rgba(10,14,23,0.92)' : 'rgba(255,255,255,0.92)',
          borderColor: isDark ? '#1F2937' : '#E5E7EB',
          borderWidth: 1,
          textStyle: { color: isDark ? '#E0E0E0' : '#111827' },
        },
        grid: { left: '6%', right: '4%', top: '12%', bottom: '8%' },
        xAxis: {
          type: 'category',
          data: dates,
          axisLine: { lineStyle: { color: isDark ? '#1F2937' : '#D1D5DB' } },
          axisLabel: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 9 },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value',
          splitLine: { lineStyle: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)', type: 'dashed' } },
          axisLabel: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
        },
        series: [
          {
            name: '买入持有',
            type: 'line',
            data: norm,
            lineStyle: { color: '#6B7280', width: 1, type: 'dashed' },
            symbol: 'none',
          },
          {
            name: '策略净值',
            type: 'line',
            data: strategyNav,
            lineStyle: { color: '#00E5FF', width: 2 },
            areaStyle: { color: 'rgba(0, 229, 255, 0.1)' },
            symbol: 'none',
          },
        ],
        legend: {
          data: ['买入持有', '策略净值'],
          textStyle: { color: isDark ? '#9CA3AF' : '#4B5563', fontSize: 11 },
          top: 0,
          right: '2%',
          itemWidth: 16,
          itemHeight: 8,
        },
      });
      equityInstance.current.resize();
    }

    // ===== 回撤曲线 =====
    if (drawdownChartRef.current) {
      if (drawdownInstance.current) drawdownInstance.current.dispose();
      drawdownInstance.current = echarts.init(drawdownChartRef.current, isDark ? 'dark' : 'light');

      const dates = ohlc.map(d => d.date);
      const closes = ohlc.map(d => d.close);
      const drawdowns: number[] = [];
      let peak = closes[0] || 0;
      for (const price of closes) {
        if (price > peak) peak = price;
        drawdowns.push((price - peak) / peak);
      }

      const maxDD = Math.min(...drawdowns);

      drawdownInstance.current.setOption({
        backgroundColor: 'transparent',
        tooltip: {
          trigger: 'axis',
          backgroundColor: isDark ? 'rgba(10,14,23,0.92)' : 'rgba(255,255,255,0.92)',
          borderColor: isDark ? '#1F2937' : '#E5E7EB',
          borderWidth: 1,
          textStyle: { color: isDark ? '#E0E0E0' : '#111827' },
          formatter: (params: any) => {
            const val = params[0]?.value;
            return `${params[0]?.axisValue}<br>回撤: ${(val * 100).toFixed(2)}%`;
          },
        },
        grid: { left: '6%', right: '4%', top: '12%', bottom: '8%' },
        xAxis: {
          type: 'category',
          data: dates,
          axisLine: { lineStyle: { color: isDark ? '#1F2937' : '#D1D5DB' } },
          axisLabel: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 9 },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value',
          splitLine: { lineStyle: { color: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)', type: 'dashed' } },
          axisLabel: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 10, formatter: (v: number) => (v * 100).toFixed(1) + '%' },
          min: Math.min(0, maxDD * 1.1),
          max: 0.01,
        },
        series: [
          {
            name: '回撤',
            type: 'line',
            data: drawdowns,
            lineStyle: { color: '#FF0050', width: 1.5 },
            areaStyle: { color: 'rgba(255, 0, 80, 0.2)' },
            symbol: 'none',
            markPoint: {
              data: [
                {
                  type: 'min',
                  name: '最大回撤',
                  symbol: 'pin',
                  symbolSize: 40,
                  itemStyle: { color: '#FF0050' },
                  label: {
                    formatter: (p: any) => (p.value * 100).toFixed(1) + '%',
                    color: '#FFFFFF',
                  },
                },
              ],
            },
          },
        ],
        legend: { show: false },
      });
      drawdownInstance.current.resize();
    }

    const handleResize = () => {
      equityInstance.current?.resize();
      drawdownInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      equityInstance.current?.dispose();
      drawdownInstance.current?.dispose();
    };
  }, [ohlc, probs, isDark]);

  // 计算统计数据（基于价格）
  const returns = ohlc.map((d, i) => (i > 0 ? (d.close - ohlc[i - 1].close) / ohlc[i - 1].close : 0));
  const totalRet = returns.reduce((a, b) => a * (1 + b), 1) - 1;
  const avgRet = returns.reduce((a, b) => a + b, 0) / (returns.length || 1);
  const stdRet = Math.sqrt(returns.reduce((a, b) => a + (b - avgRet) ** 2, 0) / ((returns.length || 1) - 1 || 1));
  const sharpe = stdRet === 0 ? 0 : (avgRet / stdRet) * Math.sqrt(252);
  let maxDD = 0;
  let peak = 1;
  for (const r of returns) {
    peak = Math.max(peak, peak * (1 + r));
    maxDD = Math.min(maxDD, (peak * (1 + r) - peak) / peak);
  }

  const bg = isDark ? 'rgba(17,24,39,0.6)' : 'rgba(255,255,255,0.8)';
  const border = isDark ? '#1F2937' : '#E5E7EB';
  const textColor = isDark ? '#9CA3AF' : '#4B5563';
  const valueColor = isDark ? '#E0E0E0' : '#111827';

  return (
    <div
      style={{
        background: bg,
        borderRadius: '12px',
        padding: '16px',
        border: `1px solid ${border}`,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h4 style={{ color: textColor, margin: 0, fontSize: '14px' }}>📊 绩效仪表盘 — {stockName}</h4>
        <div style={{ display: 'flex', gap: '16px', fontSize: '11px' }}>
          <span>
            总收益:{' '}
            <span style={{ color: totalRet >= 0 ? '#00F5A0' : '#FF0050' }}>{(totalRet * 100).toFixed(2)}%</span>
          </span>
          <span>
            夏普: <span style={{ color: '#00E5FF' }}>{sharpe.toFixed(3)}</span>
          </span>
          <span>
            最大回撤: <span style={{ color: '#FF0050' }}>{(maxDD * 100).toFixed(2)}%</span>
          </span>
          <span>
            交易天数: <span style={{ color: '#F59E0B' }}>{ohlc.length}</span>
          </span>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div ref={equityChartRef} style={{ height: '280px', minHeight: '200px' }} />
        <div ref={drawdownChartRef} style={{ height: '280px', minHeight: '200px' }} />
      </div>
    </div>
  );
};

export default PerformanceDashboard;