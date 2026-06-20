import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';

const TickerTape: React.FC = () => {
  const { stocks, selected, setSelected, theme } = useApp();
  const [prices, setPrices] = useState<Record<string, number>>({});
  const [changes, setChanges] = useState<Record<string, number>>({});
  const isDark = theme === 'dark';

  useEffect(() => {
    const initialPrices: Record<string, number> = {};
    const initialChanges: Record<string, number> = {};
    stocks.forEach((s, i) => {
      const basePrice = 50 + Math.random() * 150;
      initialPrices[s.stock] = basePrice;
      initialChanges[s.stock] = (Math.random() - 0.5) * 2;
    });
    setPrices(initialPrices);
    setChanges(initialChanges);

    const interval = setInterval(() => {
      setPrices(prev => {
        const newPrices = { ...prev };
        Object.keys(newPrices).forEach(key => {
          const change = (Math.random() - 0.5) * 1.5;
          newPrices[key] = Math.max(10, newPrices[key] * (1 + change / 100));
          setChanges(prevChanges => ({ ...prevChanges, [key]: change }));
        });
        return newPrices;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [stocks]);

  const handleClick = (stock: string) => {
    if (stock !== selected) setSelected(stock);
  };

  if (stocks.length === 0) return null;

  return (
    <div style={{
      background: isDark ? '#111827' : '#E5E7EB',
      borderBottom: `1px solid ${isDark ? '#1F2937' : '#D1D5DB'}`,
      padding: '6px 0',
      overflow: 'hidden',
      whiteSpace: 'nowrap',
      borderRadius: '8px',
      marginBottom: '12px',
      cursor: 'pointer',
    }}>
      <div style={{
        display: 'flex',
        animation: 'scroll 50s linear infinite',
      }}>
        {[...stocks, ...stocks, ...stocks].map((s, idx) => {
          const price = prices[s.stock] || 100;
          const change = changes[s.stock] || 0;
          const color = change > 0 ? '#00F5A0' : change < 0 ? '#FF0050' : '#F59E0B';
          const isSelected = s.stock === selected;
          return (
            <span
              key={idx}
              onClick={() => handleClick(s.stock)}
              style={{
                margin: '0 25px',
                fontSize: '13px',
                color: isDark ? '#D1D5DB' : '#374151',
                padding: '2px 8px',
                borderRadius: '4px',
                background: isSelected ? 'rgba(0,229,255,0.1)' : 'transparent',
                border: isSelected ? `1px solid ${isDark ? 'rgba(0,229,255,0.2)' : 'rgba(0,229,255,0.4)'}` : '1px solid transparent',
                transition: 'all 0.2s',
                cursor: 'pointer',
              }}
            >
              <span style={{ fontWeight: isSelected ? 700 : 600 }}>{s.stock.slice(0, 10)}</span>
              <span style={{ margin: '0 8px', fontWeight: 700, color: isDark ? '#FFFFFF' : '#111827' }}>{price.toFixed(2)}</span>
              <span style={{ color }}>{change > 0 ? '+' : ''}{change.toFixed(2)}%</span>
              {isSelected && <span style={{ marginLeft: '6px', color: '#00E5FF', fontSize: '10px' }}>◀</span>}
            </span>
          );
        })}
      </div>
      <style>
        {`
          @keyframes scroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(-33.33%); }
          }
        `}
      </style>
    </div>
  );
};

export default TickerTape;