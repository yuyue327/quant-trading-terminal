import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

interface MonthlyHeatmapProps {
  ohlc: any[];
  probs: any[];
}

const MonthlyHeatmap: React.FC<MonthlyHeatmapProps> = ({ ohlc, probs }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!chartRef.current || ohlc.length === 0) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current, 'dark');

    // --- 计算月度收益率 ---
    // 将日期转换为月份，计算每月最后一天的收盘价相对于上个月最后一天的涨跌幅
    const monthlyData: Record<string, { month: string; year: string; return: number }> = {};

    // 按月分组
    const byMonth: Record<string, number[]> = {};
    ohlc.forEach(d => {
      const date = new Date(d.date);
      const key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      if (!byMonth[key]) byMonth[key] = [];
      byMonth[key].push(d.close);
    });

    const months = Object.keys(byMonth).sort();
    const yearMap: Record<string, Record<string, number>> = {};

    // 计算每月收益率
    for (let i = 0; i < months.length; i++) {
      const month = months[i];
      const prices = byMonth[month];
      const lastPrice = prices[prices.length - 1];
      const firstPrice = prices[0];
      const ret = (lastPrice - firstPrice) / firstPrice;

      const [year, monthNum] = month.split('-');
      if (!yearMap[year]) yearMap[year] = {};
      yearMap[year][monthNum] = ret;
    }

    // 构建热力图数据
    const years = Object.keys(yearMap).sort();
    const monthNums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'];
    const data: any[] = [];
    years.forEach(year => {
      monthNums.forEach(month => {
        const value = yearMap[year]?.[month];
        if (value !== undefined) {
          data.push([month, year, value]);
        }
      });
    });

    const option: echarts.EChartsOption = {
      tooltip: {
        formatter: (params: any) => {
          const [month, year, value] = params.data;
          return `${year}年${month}月<br>收益率: ${(value * 100).toFixed(2)}%`;
        },
      },
      grid: { left: '10%', right: '5%', top: '10%', bottom: '10%' },
      xAxis: {
        type: 'category',
        data: monthNums.map(m => `${m}月`),
        splitArea: { show: true },
        axisLabel: { color: '#6B7280', fontSize: 10 },
      },
      yAxis: {
        type: 'category',
        data: years,
        splitArea: { show: true },
        axisLabel: { color: '#6B7280', fontSize: 10 },
      },
      visualMap: {
        min: -0.3,
        max: 0.3,
        calculable: true,
        inRange: {
          color: ['#FF0050', '#FF6B6B', '#FFE66D', '#00F5A0', '#00C9A7'],
        },
        textStyle: { color: '#9CA3AF', fontSize: 10 },
        left: 'right',
        bottom: 10,
      },
      series: [{
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          formatter: (params: any) => {
            const value = params.data[2];
            return `${(value * 100).toFixed(0)}%`;
          },
          fontSize: 9,
          color: '#E0E0E0',
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0,0,0,0.5)',
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
    };
  }, [ohlc, probs]);

  return <div ref={chartRef} style={{ width: '100%', height: '320px' }} />;
};

export default MonthlyHeatmap;