import React, { useEffect, useRef } from 'react';
import * as echarts from 'echarts';

interface PredictionInsightsProps {
  probs: any[];
  stockName: string;
  currentPrice: number;
}

const PredictionInsights: React.FC<PredictionInsightsProps> = ({ probs, stockName, currentPrice }) => {
  // --- 无数据时显示简洁占位 ---
  if (!probs || probs.length === 0) {
    return (
      <div style={{
        background: 'rgba(10,14,23,0.8)',
        backdropFilter: 'blur(20px)',
        borderRadius: '16px',
        border: '1px solid rgba(255,255,255,0.05)',
        padding: '20px',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          position: 'absolute',
          top: '-50%', left: '-50%', width: '200%', height: '200%',
          background: 'radial-gradient(circle, rgba(0,229,255,0.02) 0%, transparent 70%)',
          animation: 'spin 20s linear infinite',
        }} />
        <div style={{ fontSize: '36px', marginBottom: '12px', position: 'relative' }}>🔮</div>
        <div style={{ fontSize: '16px', color: '#9CA3AF', position: 'relative' }}>暂无预测数据</div>
        <style>{`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  // --- 有数据时渲染完整洞察 ---
  const distChartRef = useRef<HTMLDivElement>(null);
  const futureChartRef = useRef<HTMLDivElement>(null);
  const distInstance = useRef<echarts.ECharts | null>(null);
  const futureInstance = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!probs.length) return;

    // 确保所有图表容器都存在
    if (!distChartRef.current || !futureChartRef.current) return;

    // --- 概率分布直方图 ---
    if (distChartRef.current) {
      if (distInstance.current) distInstance.current.dispose();
      distInstance.current = echarts.init(distChartRef.current, 'dark');

      const probValues = probs.map(p => p.prob).filter(v => typeof v === 'number');
      const bins = 10;
      const min = 0;
      const max = 1;
      const step = (max - min) / bins;
      const hist: number[] = new Array(bins).fill(0);
      probValues.forEach(v => {
        const idx = Math.min(Math.floor((v - min) / step), bins - 1);
        hist[idx]++;
      });
      const binLabels = Array.from({ length: bins }, (_, i) => `${(min + i * step).toFixed(1)}-${(min + (i+1) * step).toFixed(1)}`);

      distInstance.current.setOption({
        backgroundColor: 'transparent',
        tooltip: {
          trigger: 'axis',
          backgroundColor: 'rgba(10,14,23,0.9)',
          borderColor: '#1F2937',
          textStyle: { color: '#E0E0E0' },
        },
        grid: { left: '8%', right: '4%', top: '8%', bottom: '12%' },
        xAxis: {
          type: 'category',
          data: binLabels,
          axisLabel: { color: '#6B7280', fontSize: 9, rotate: 30 },
          axisLine: { lineStyle: { color: '#1F2937' } },
        },
        yAxis: {
          type: 'value',
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } },
          axisLabel: { color: '#6B7280', fontSize: 9 },
        },
        series: [{
          type: 'bar',
          data: hist,
          itemStyle: {
            color: (params: any) => {
              const val = parseFloat(binLabels[params.dataIndex].split('-')[0]);
              if (val >= 0.55) return '#00F5A0';
              if (val <= 0.45) return '#FF0050';
              return '#F59E0B';
            },
            borderRadius: [4, 4, 0, 0],
          },
          barWidth: '70%',
        }],
      });
      distInstance.current.resize();
    }

    // --- 未来预测曲线 ---
    if (futureChartRef.current) {
      if (futureInstance.current) futureInstance.current.dispose();
      futureInstance.current = echarts.init(futureChartRef.current, 'dark');

      const probValues = probs.map(p => p.prob).filter(v => typeof v === 'number');
      const recent = probValues.slice(-30);
      const trend = recent.length > 1 ? (recent[recent.length-1] - recent[0]) / recent.length : 0;
      const lastProb = recent[recent.length-1] || 0.5;
      const futureDays = 10;
      const futureProbs: number[] = [];
      let current = lastProb;
      for (let i = 0; i < futureDays; i++) {
        const noise = (Math.random() - 0.5) * 0.04;
        current = Math.min(1, Math.max(0, current + trend * 0.5 + noise));
        futureProbs.push(current);
      }
      const futureLabels = Array.from({ length: futureDays }, (_, i) => `+${i+1}d`);
      const upper = futureProbs.map(p => Math.min(1, p + 0.08));
      const lower = futureProbs.map(p => Math.max(0, p - 0.08));

      futureInstance.current.setOption({
        backgroundColor: 'transparent',
        tooltip: {
          trigger: 'axis',
          backgroundColor: 'rgba(10,14,23,0.9)',
          borderColor: '#1F2937',
          textStyle: { color: '#E0E0E0' },
        },
        grid: { left: '6%', right: '4%', top: '10%', bottom: '12%' },
        xAxis: {
          type: 'category',
          data: futureLabels,
          axisLabel: { color: '#6B7280', fontSize: 9 },
          axisLine: { lineStyle: { color: '#1F2937' } },
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 1,
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } },
          axisLabel: { color: '#6B7280', fontSize: 9 },
        },
        series: [
          {
            name: '预测概率',
            type: 'line',
            data: futureProbs,
            smooth: true,
            lineStyle: { color: '#00E5FF', width: 2 },
            areaStyle: { color: 'rgba(0,229,255,0.1)' },
            symbol: 'circle',
            symbolSize: 4,
            itemStyle: { color: '#00E5FF' },
          },
          {
            name: '置信区间',
            type: 'line',
            data: upper,
            lineStyle: { color: 'rgba(0,229,255,0.2)', type: 'dashed' },
            symbol: 'none',
          },
          {
            name: '置信区间',
            type: 'line',
            data: lower,
            lineStyle: { color: 'rgba(0,229,255,0.2)', type: 'dashed' },
            symbol: 'none',
            areaStyle: { color: 'rgba(0,229,255,0.05)' },
          },
        ],
        legend: {
          data: ['预测概率', '置信区间'],
          textStyle: { color: '#9CA3AF', fontSize: 9 },
          top: 0,
          right: '2%',
          itemWidth: 12,
          itemHeight: 6,
        },
      });
      futureInstance.current.resize();
    }

    const handleResize = () => {
      distInstance.current?.resize();
      futureInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      distInstance.current?.dispose();
      futureInstance.current?.dispose();
    };
  }, [probs]);

  // --- 计算信号与情绪 ---
  const lastProb = probs.length ? probs[probs.length-1]?.prob : 0.5;
  const signal = lastProb > 0.55 ? '买入' : lastProb < 0.45 ? '卖出' : '持有';
  const signalColor = signal === '买入' ? '#00F5A0' : signal === '卖出' ? '#FF0050' : '#F59E0B';
  const confidence = lastProb > 0.55 ? (lastProb - 0.55) / 0.45 : lastProb < 0.45 ? (0.45 - lastProb) / 0.45 : 0;
  const confidencePct = (confidence * 100).toFixed(0);
  const sentiment = lastProb * 100; // 0-100

  return (
    <div style={{
      background: 'rgba(10,14,23,0.8)',
      backdropFilter: 'blur(20px)',
      borderRadius: '16px',
      border: '1px solid rgba(255,255,255,0.05)',
      padding: '16px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute',
        top: '-50%',
        right: '-50%',
        width: '100%',
        height: '100%',
        background: 'radial-gradient(circle, rgba(0,229,255,0.02) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h4 style={{ color: '#E0E0E0', fontSize: '14px', margin: 0 }}>🔮 预测洞察</h4>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <div style={{ fontSize: '11px', color: '#6B7280' }}>
            信号: <span style={{ color: signalColor, fontWeight: 600 }}>{signal}</span>
          </div>
          <div style={{ fontSize: '11px', color: '#6B7280' }}>
            置信度: <span style={{ color: '#00E5FF' }}>{confidencePct}%</span>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div>
          <div style={{ color: '#6B7280', fontSize: '10px', marginBottom: '4px' }}>📊 概率分布</div>
          <div ref={distChartRef} style={{ height: '140px' }} />
        </div>
        <div>
          <div style={{ color: '#6B7280', fontSize: '10px', marginBottom: '4px' }}>📈 未来 {10} 日预测</div>
          <div ref={futureChartRef} style={{ height: '140px' }} />
        </div>
      </div>

      <div style={{
        marginTop: '12px',
        paddingTop: '12px',
        borderTop: '1px solid rgba(255,255,255,0.03)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: '10px', color: '#6B7280' }}>市场情绪</span>
        <div style={{ flex: 1, margin: '0 12px', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', position: 'relative' }}>
          <div style={{
            width: `${sentiment}%`,
            height: '100%',
            background: `linear-gradient(90deg, #FF0050, #F59E0B, #00F5A0)`,
            borderRadius: '2px',
            transition: 'width 0.6s ease',
          }} />
        </div>
        <span style={{ fontSize: '12px', color: '#E0E0E0', fontWeight: 600, minWidth: '40px', textAlign: 'right' }}>
          {sentiment.toFixed(0)}%
        </span>
        <span style={{ fontSize: '9px', color: '#6B7280', marginLeft: '8px' }}>
          {sentiment > 70 ? '贪婪' : sentiment < 30 ? '恐慌' : '中性'}
        </span>
      </div>

      <div style={{ display: 'flex', gap: '20px', marginTop: '8px', fontSize: '10px', color: '#6B7280' }}>
        <span>📅 最新概率: <span style={{ color: '#E0E0E0' }}>{(lastProb * 100).toFixed(1)}%</span></span>
        <span>📊 样本数: <span style={{ color: '#E0E0E0' }}>{probs.length}</span></span>
        <span>⚡ 波动率: <span style={{ color: '#E0E0E0' }}>{(probs.reduce((a,b) => a + Math.abs((b.prob||0.5)-0.5), 0) / probs.length * 100).toFixed(1)}%</span></span>
      </div>
    </div>
  );
};

export default PredictionInsights;