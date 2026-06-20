import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';

const RecentTrades: React.FC = () => {
  const { theme } = useApp();
  const isDark = theme === 'dark';
  const [trades, setTrades] = useState<any[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem('pro_orders');
    if (saved) {
      try {
        const orders = JSON.parse(saved);
        setTrades(orders.slice(0, 10));
      } catch (e) {}
    }
  }, []);

  useEffect(() => {
    const handleStorage = () => {
      const saved = localStorage.getItem('pro_orders');
      if (saved) {
        try {
          const orders = JSON.parse(saved);
          setTrades(orders.slice(0, 10));
        } catch (e) {}
      }
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  const bg = isDark ? 'rgba(17,24,39,0.6)' : 'rgba(255,255,255,0.7)';
  const borderColor = isDark ? '#1F2937' : '#E5E7EB';
  const textColor = isDark ? '#9CA3AF' : '#4B5563';
  const textPrimary = isDark ? '#E0E0E0' : '#111827';

  if (trades.length === 0) {
    return (
      <div style={{
        background: bg,
        borderRadius: '12px',
        padding: '16px',
        border: `1px solid ${borderColor}`,
        textAlign: 'center',
        color: textColor,
        fontSize: '13px',
      }}>
        📋 暂无交易记录
      </div>
    );
  }

  return (
    <div
      className="recent-trades"
      style={{
        background: bg,
        borderRadius: '12px',
        padding: '12px',
        border: `1px solid ${borderColor}`,
        maxHeight: '220px',
        overflow: 'auto',
      }}
    >
      <div style={{ color: textColor, fontSize: '11px', marginBottom: '8px', fontWeight: 600 }}>
        📋 最近交易
      </div>
      {trades.map((t: any, idx: number) => (
        <div
          key={idx}
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            padding: '4px 0',
            borderBottom: `1px solid ${isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)'}`,
            fontSize: '11px',
            color: textPrimary,
            flexWrap: 'wrap',
            gap: '2px',
          }}
        >
          <span>{t.stock?.slice(0, 10) || '--'}</span>
          <span style={{ color: t.side === 'buy' ? '#00F5A0' : '#FF0050' }}>
            {t.side === 'buy' ? '🔼 买' : '🔽 卖'}
          </span>
          <span>{t.shares || 0}股</span>
          <span>¥{(t.price || 0).toFixed(2)}</span>
          <span style={{ color: textColor, fontSize: '9px' }}>
            {t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : ''}
          </span>
        </div>
      ))}
    </div>
  );
};

export default RecentTrades;