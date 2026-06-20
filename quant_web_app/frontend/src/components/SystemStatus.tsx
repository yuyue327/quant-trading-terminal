import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';

const SystemStatus: React.FC = () => {
  const { isConnected, latency, stocks, selected, theme } = useApp();
  const [uptime, setUptime] = useState(0);
  const isDark = theme === 'dark';

  useEffect(() => {
    const interval = setInterval(() => setUptime(prev => prev + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '16px',
      padding: '4px 16px',
      background: isDark ? 'rgba(17,24,39,0.6)' : 'rgba(255,255,255,0.6)',
      borderRadius: '20px',
      border: `1px solid ${isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)'}`,
      fontSize: '11px',
      color: '#6B7280',
      flexWrap: 'wrap',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          background: isConnected ? '#00F5A0' : '#FF0050',
          boxShadow: isConnected ? '0 0 10px rgba(0,245,160,0.4)' : 'none',
          animation: isConnected ? 'pulse-dot 1.5s ease-in-out infinite' : 'none',
        }} />
        <span style={{ color: isConnected ? '#00F5A0' : '#FF0050' }}>
          {isConnected ? '在线' : '离线'}
        </span>
      </div>
      <span>延迟: <span style={{ color: latency < 30 ? '#00F5A0' : latency < 100 ? '#F59E0B' : '#FF0050' }}>{latency}ms</span></span>
      <span>运行: <span style={{ color: isDark ? '#E0E0E0' : '#111827', fontFamily: 'monospace' }}>{formatUptime(uptime)}</span></span>
      <span>标的: <span style={{ color: isDark ? '#E0E0E0' : '#111827' }}>{selected?.slice(0, 12) || '--'}</span></span>
      <span>股票池: <span style={{ color: isDark ? '#E0E0E0' : '#111827' }}>{stocks.length}</span></span>
      <style>
        {`
          @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
          }
        `}
      </style>
    </div>
  );
};

export default SystemStatus;