import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';
import { useApp } from '../context/AppContext';

const stockSectors: Record<string, string> = {
  'A_sh.600036_招商银行': '银行',
  'A_sz.000001_平安银行': '银行',
  'A_sh.600030_中信证券': '券商',
  'A_sz.000858_五粮液': '消费',
  'A_sh.600519_贵州茅台': '消费',
  'A_sz.300750_宁德时代': '新能源',
  'A_sh.600887_伊利股份': '消费',
  'A_sh.601012_隆基绿能': '新能源',
  'A_sh.601688_华泰证券': '券商',
  'A_sz.000333_美的集团': '家电',
  'A_sz.000568_泸州老窖': '消费',
  'A_sz.000651_格力电器': '家电',
  'A_sz.002142_宁波银行': '银行',
  'A_sz.002594_比亚迪': '新能源',
  'A_sz.300059_东方财富': '券商',
  'US_AAPL_AAPL': '科技',
  'US_MSFT_MSFT': '科技',
  'US_NVDA_NVDA': '科技',
};

const RiskReturnScatter: React.FC = () => {
  const { stocks, theme } = useApp();
  const isDark = theme === 'dark';
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current || stocks.length === 0) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
      chartInstance.current = null;
    }

    chartInstance.current = echarts.init(chartRef.current, isDark ? 'dark' : 'light');

    const validStocks = stocks.filter(s =>
      s.annual_return_median !== 0 &&
      s.max_drawdown_median !== 0 &&
      s.sharpe_median !== 0
    );

    if (validStocks.length === 0) {
      chartInstance.current.setOption({
        title: {
          text: '暂无足够数据',
          left: 'center',
          top: 'center',
          textStyle: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 14, fontWeight: 400 },
        },
      });
      return;
    }

    const data = validStocks.map(s => ({
      name: s.stock.slice(0, 14),
      value: [(s.max_drawdown_median || 0) * 100, (s.annual_return_median || 0) * 100],
      sharpe: s.sharpe_median || 0,
      sector: stockSectors[s.stock] || '其他',
    }));

    const sectorColors: Record<string, string> = {
      '银行': '#00E5FF',
      '券商': '#00F5A0',
      '消费': '#F59E0B',
      '新能源': '#8B5CF6',
      '科技': '#FF6B6B',
      '家电': '#FF9F43',
      '其他': '#6B7280',
    };

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        backgroundColor: isDark ? 'rgba(10,14,23,0.92)' : 'rgba(255,255,255,0.92)',
        borderColor: isDark ? '#1F2937' : '#E5E7EB',
        borderWidth: 1,
        textStyle: { color: isDark ? '#E0E0E0' : '#111827', fontSize: 12 },
        formatter: (params: any) => {
          const d = params.data;
          return `
            <div style="font-weight:600;margin-bottom:4px;">${d.name}</div>
            <div>行业: ${d.sector}</div>
            <div>年化收益: <span style="color:${d.value[1] >= 0 ? '#00F5A0' : '#FF0050'}">${d.value[1].toFixed(2)}%</span></div>
            <div>最大回撤: <span style="color:#FF6B6B">${d.value[0].toFixed(2)}%</span></div>
            <div>夏普比率: <span style="color:#00E5FF">${d.sharpe.toFixed(3)}</span></div>
          `;
        },
      },
      grid: {
        left: '12%',
        right: '8%',
        top: '12%',
        bottom: '14%',
      },
      xAxis: {
        name: '最大回撤 (%)',
        nameLocation: 'center',
        nameGap: 25,
        nameTextStyle: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 11, fontWeight: 500 },
        axisLine: { lineStyle: { color: isDark ? '#1F2937' : '#D1D5DB' } },
        axisLabel: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 10, formatter: (v: number) => v.toFixed(0) + '%' },
        splitLine: { lineStyle: { color: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)', type: 'dashed' } },
      },
      yAxis: {
        name: '年化收益 (%)',
        nameLocation: 'center',
        nameGap: 35,
        nameTextStyle: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 11, fontWeight: 500 },
        axisLine: { lineStyle: { color: isDark ? '#1F2937' : '#D1D5DB' } },
        axisLabel: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 10, formatter: (v: number) => v.toFixed(0) + '%' },
        splitLine: { lineStyle: { color: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)', type: 'dashed' } },
      },
      series: [{
        type: 'scatter',
        data: data,
        symbolSize: (val: any) => {
          const sharpe = val.sharpe || 0;
          return Math.max(12, Math.min(40, 10 + sharpe * 6));
        },
        itemStyle: {
          shadowBlur: 10,
          shadowColor: isDark ? 'rgba(0,229,255,0.15)' : 'rgba(0,229,255,0.2)',
          color: (params: any) => {
            const sector = params.data.sector || '其他';
            return sectorColors[sector] || '#6B7280';
          },
          opacity: 0.85,
        },
        label: {
          show: true,
          formatter: (params: any) => params.data.name,
          position: 'top',
          color: isDark ? '#9CA3AF' : '#4B5563',
          fontSize: 9,
          distance: 6,
        },
        emphasis: {
          scale: 1.5,
          itemStyle: {
            shadowBlur: 20,
            shadowColor: isDark ? 'rgba(0,229,255,0.4)' : 'rgba(0,229,255,0.5)',
            opacity: 1,
          },
        },
      }],
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, [stocks, isDark]);

  const bg = isDark ? 'rgba(17,24,39,0.4)' : 'rgba(255,255,255,0.7)';
  const borderColor = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)';
  const textColor = isDark ? '#9CA3AF' : '#4B5563';

  return (
    <div style={{
      background: bg,
      borderRadius: '16px',
      padding: '16px',
      border: `1px solid ${borderColor}`,
      backdropFilter: 'blur(10px)',
      height: '100%',
    }}>
      <div style={{
        color: textColor,
        fontSize: '12px',
        marginBottom: '8px',
        fontWeight: 500,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span>📈 风险收益散点图</span>
        <span style={{ fontSize: '10px', color: isDark ? '#6B7280' : '#4B5563' }}>气泡大小 = 夏普比率</span>
      </div>
      <div ref={chartRef} style={{ width: '100%', height: '300px', minHeight: '200px' }} />
    </div>
  );
};

export default RiskReturnScatter;