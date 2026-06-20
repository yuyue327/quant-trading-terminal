import React, { useEffect, useRef, useState } from 'react';
import { useApp } from '../context/AppContext';
import AnimatedCounter from './AnimatedCounter';

const HeroStats: React.FC = () => {
  const { stocks, theme } = useApp();
  const isDark = theme === 'dark';
  const [counts, setCounts] = useState({
    totalReturn: 0,
    avgSharpe: 0,
    winRate: 0,
    totalTrades: 0,
  });
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (stocks.length === 0) return;
    const valid = stocks.filter(s => s.sharpe_median !== 0);
    const avgSharpe = valid.reduce((sum, s) => sum + s.sharpe_median, 0) / (valid.length || 1);
    const avgWinRate = valid.reduce((sum, s) => sum + s.win_rate_median, 0) / (valid.length || 1);
    const totalTrades = valid.reduce((sum, s) => sum + (s.total_trades_median || 0), 0);
    const firstValid = valid.find(s => s.annual_return_median !== 0);
    setCounts({
      totalReturn: firstValid ? firstValid.annual_return_median * 100 : 0,
      avgSharpe: avgSharpe,
      winRate: avgWinRate * 100,
      totalTrades: totalTrades,
    });
  }, [stocks]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setIsVisible(true);
      },
      { threshold: 0.3 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  const stats = [
    { label: '总收益', value: counts.totalReturn, suffix: '%', color: '#00F5A0' },
    { label: '平均夏普', value: counts.avgSharpe, suffix: '', color: '#00E5FF' },
    { label: '平均胜率', value: counts.winRate, suffix: '%', color: '#F59E0B' },
    { label: '总交易', value: counts.totalTrades, suffix: '', color: '#8B5CF6' },
  ];

  const bg = isDark ? 'rgba(17,24,39,0.4)' : 'rgba(255,255,255,0.7)';
  const borderColor = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)';
  const textColor = isDark ? '#6B7280' : '#4B5563';
  const dividerColor = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)';

  return (
    <div
      ref={ref}
      className="hero-stats-grid"
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '1px',
        background: borderColor,
        borderRadius: '16px',
        overflow: 'hidden',
        marginBottom: '24px',
        border: `1px solid ${borderColor}`,
      }}
    >
      {stats.map((stat, idx) => (
        <div
          key={idx}
          style={{
            background: bg,
            padding: '20px 24px',
            textAlign: 'center',
            backdropFilter: 'blur(10px)',
            borderRight: idx < 3 ? `1px solid ${dividerColor}` : 'none',
          }}
        >
          <div
            className="stat-value"
            style={{
              fontSize: '32px',
              fontWeight: 700,
              color: stat.color,
              fontFamily: 'Inter, sans-serif',
              letterSpacing: '-0.5px',
              transition: 'all 0.8s ease',
              transform: isVisible ? 'scale(1)' : 'scale(0.8)',
              opacity: isVisible ? 1 : 0,
            }}
          >
            {isVisible ? (
              <AnimatedCounter
                value={stat.value}
                duration={1500}
                decimals={stat.suffix === '%' ? 1 : 0}
                suffix={stat.suffix}
              />
            ) : '0'}
          </div>
          <div
            className="stat-label"
            style={{
              fontSize: '12px',
              color: textColor,
              marginTop: '4px',
              letterSpacing: '0.5px',
              textTransform: 'uppercase',
            }}
          >
            {stat.label}
          </div>
        </div>
      ))}
    </div>
  );
};

export default HeroStats;