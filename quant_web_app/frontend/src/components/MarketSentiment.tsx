import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import { useApp } from '../context/AppContext';

interface MarketSentimentProps {
  className?: string;
}

const MarketSentiment: React.FC<MarketSentimentProps> = ({ className }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const { ohlc, theme } = useApp();
  const isDark = theme === 'dark';
  const [sentiment, setSentiment] = useState(50);

  useEffect(() => {
    if (ohlc && ohlc.length > 20) {
      const recent = ohlc.slice(-20);
      const firstClose = recent[0]?.close || 1;
      const lastClose = recent[recent.length - 1]?.close || 1;
      const change = (lastClose - firstClose) / firstClose;
      const mapped = Math.min(100, Math.max(0, 50 + change * 100));
      setSentiment(mapped);
    } else {
      const interval = setInterval(() => {
        setSentiment(prev => Math.min(100, Math.max(0, prev + (Math.random() - 0.5) * 3)));
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [ohlc]);

  useEffect(() => {
    if (!chartRef.current) return;

    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    chartInstance.current = echarts.init(chartRef.current, isDark ? 'dark' : 'light');

    const isGreedy = sentiment > 60;
    const isFearful = sentiment < 40;

    const option: echarts.EChartsOption = {
      backgroundColor: 'transparent',
      series: [{
        type: 'gauge',
        startAngle: 220,
        endAngle: -40,
        min: 0,
        max: 100,
        splitNumber: 10,
        radius: '75%',
        center: ['50%', '55%'],
        progress: {
          show: true,
          width: 10,
          roundCap: true,
          itemStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 1, y2: 0,
              colorStops: [
                { offset: 0, color: '#FF0050' },
                { offset: 0.4, color: '#F59E0B' },
                { offset: 0.6, color: '#F59E0B' },
                { offset: 1, color: '#00F5A0' },
              ]
            }
          }
        },
        pointer: {
          icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
          length: '45%',
          width: 4,
          offsetCenter: [0, '-5%'],
          itemStyle: { color: isDark ? '#E0E0E0' : '#111827' },
        },
        axisLine: {
          lineStyle: {
            width: 10,
            color: [
              [0.3, '#FF0050'],
              [0.5, '#F59E0B'],
              [0.7, '#F59E0B'],
              [1, '#00F5A0']
            ]
          }
        },
        axisTick: {
          distance: -8,
          length: 4,
          lineStyle: { color: isDark ? '#6B7280' : '#4B5563', width: 1 }
        },
        splitLine: {
          distance: -12,
          length: 10,
          lineStyle: { color: isDark ? '#6B7280' : '#4B5563', width: 2 }
        },
        axisLabel: {
          color: isDark ? '#6B7280' : '#4B5563',
          fontSize: 8,
          distance: 2,
        },
        detail: {
          valueAnimation: true,
          formatter: '{value}',
          color: isDark ? '#E0E0E0' : '#111827',
          fontSize: 14,
          fontWeight: 700,
          offsetCenter: [0, '25%'],
        },
        title: {
          offsetCenter: [0, '42%'],
          fontSize: 9,
          color: isDark ? '#6B7280' : '#4B5563',
        },
        data: [{
          value: Math.round(sentiment),
          name: isGreedy ? '🤑 贪婪' : isFearful ? '😨 恐慌' : '⚖️ 中性'
        }]
      }],
    };

    chartInstance.current.setOption(option);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, [sentiment, isDark]);

  const bg = isDark ? 'rgba(17,24,39,0.6)' : 'rgba(255,255,255,0.7)';
  const borderColor = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)';
  const textColor = isDark ? '#6B7280' : '#4B5563';

  return (
    <div
      className={className}
      style={{
        background: bg,
        borderRadius: '12px',
        padding: '6px 10px',
        border: `1px solid ${borderColor}`,
        marginBottom: '10px',
      }}
    >
      <div style={{ color: textColor, fontSize: '9px', marginBottom: '0px', textAlign: 'center' }}>
        🧠 市场情绪
      </div>
      <div ref={chartRef} style={{ width: '100%', height: '140px' }} />
    </div>
  );
};

export default MarketSentiment;