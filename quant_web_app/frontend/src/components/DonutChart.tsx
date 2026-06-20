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

const DonutChart: React.FC = () => {
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

    const sectorMap: Record<string, { count: number; totalSharpe: number }> = {};
    stocks.forEach(s => {
      const sector = stockSectors[s.stock] || '其他';
      if (!sectorMap[sector]) sectorMap[sector] = { count: 0, totalSharpe: 0 };
      sectorMap[sector].count += 1;
      sectorMap[sector].totalSharpe += s.sharpe_median || 0;
    });

    const data = Object.entries(sectorMap)
      .filter(([_, val]) => val.count > 0)
      .map(([name, val]) => ({
        name,
        value: Number((val.totalSharpe / (val.count || 1)).toFixed(3)),
      }));

    if (data.length === 0) {
      chartInstance.current.setOption({
        title: {
          text: '暂无数据',
          left: 'center',
          top: 'center',
          textStyle: { color: isDark ? '#6B7280' : '#4B5563', fontSize: 14, fontWeight: 400 },
        },
      });
      return;
    }

    const colors = ['#00E5FF', '#00F5A0', '#F59E0B', '#8B5CF6', '#FF6B6B', '#FF9F43', '#6B7280'];

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        backgroundColor: isDark ? 'rgba(10,14,23,0.92)' : 'rgba(255,255,255,0.92)',
        borderColor: isDark ? '#1F2937' : '#E5E7EB',
        borderWidth: 1,
        textStyle: { color: isDark ? '#E0E0E0' : '#111827' },
        formatter: (params: any) => {
          return `<strong>${params.name}</strong><br/>平均夏普: ${params.value.toFixed(3)}<br/>占比: ${params.percent}%`;
        },
      },
      series: [{
        type: 'pie',
        radius: ['55%', '75%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 6,
          borderColor: isDark ? '#0A0E17' : '#FFFFFF',
          borderWidth: 2,
        },
        label: {
          show: true,
          color: isDark ? '#9CA3AF' : '#4B5563',
          fontSize: 11,
          formatter: '{b}\n{d}%',
        },
        labelLine: {
          lineStyle: { color: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)' },
        },
        emphasis: {
          scale: true,
          scaleSize: 8,
        },
        data: data.map((d, i) => ({
          ...d,
          itemStyle: { color: colors[i % colors.length] },
        })),
        animationDuration: 1000,
        animationEasing: 'cubicOut',
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
      <div style={{ color: textColor, fontSize: '12px', marginBottom: '8px', fontWeight: 500 }}>
        📊 行业收益归因
      </div>
      <div ref={chartRef} style={{ width: '100%', height: '240px', minHeight: '180px' }} />
    </div>
  );
};

export default DonutChart;