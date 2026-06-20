import React from 'react';
import { useApp } from '../context/AppContext';

const QuickActions: React.FC = () => {
  const { selected, setSelected, stocks } = useApp();

  const quickStocks = stocks.slice(0, 6);

  return (
    <div style={{
      display: 'flex',
      gap: '8px',
      flexWrap: 'wrap',
      marginBottom: '12px',
    }}>
      {quickStocks.map(s => (
        <button
          key={s.stock}
          onClick={() => setSelected(s.stock)}
          style={{
            background: selected === s.stock ? 'rgba(0,229,255,0.15)' : 'rgba(255,255,255,0.03)',
            border: selected === s.stock ? '1px solid rgba(0,229,255,0.3)' : '1px solid #1F2937',
            borderRadius: '6px',
            padding: '4px 12px',
            color: selected === s.stock ? '#00E5FF' : '#9CA3AF',
            fontSize: '11px',
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
        >
          {s.stock.slice(0, 10)}
        </button>
      ))}
    </div>
  );
};

export default QuickActions;