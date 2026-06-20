import React, { useEffect } from 'react';
import { useApp } from '../context/AppContext';

const NotificationSystem: React.FC = () => {
  const { notifications, removeNotification, addNotification, selected } = useApp();

  // 模拟系统通知
  useEffect(() => {
    const messages = [
      { msg: '📈 价格突破 52 周新高', type: 'success' as const },
      { msg: '📊 预测信号：强烈买入', type: 'info' as const },
      { msg: '⚠️ 波动率异常上升', type: 'warning' as const },
      { msg: '🔔 新交易信号出现', type: 'info' as const },
      { msg: '📉 价格触及止损线', type: 'error' as const },
    ];

    const interval = setInterval(() => {
      if (Math.random() > 0.6) {
        const idx = Math.floor(Math.random() * messages.length);
        const { msg, type } = messages[idx];
        addNotification(`${selected?.slice(0, 12) || '系统'}: ${msg}`, type);
      }
    }, 8000);

    // 启动时立即发一条
    setTimeout(() => {
      addNotification(`🟢 系统已连接 · 正在监控 ${selected?.slice(0, 12) || '标的'}`, 'success');
    }, 1000);

    return () => clearInterval(interval);
  }, [selected, addNotification]);

  if (notifications.length === 0) return null;

  const getTypeStyle = (type: string) => {
    switch (type) {
      case 'success': return { borderColor: '#00F5A0', bg: 'rgba(0,245,160,0.1)', icon: '✅' };
      case 'warning': return { borderColor: '#F59E0B', bg: 'rgba(245,158,11,0.1)', icon: '⚠️' };
      case 'error': return { borderColor: '#FF0050', bg: 'rgba(255,0,80,0.1)', icon: '🚨' };
      default: return { borderColor: '#00E5FF', bg: 'rgba(0,229,255,0.1)', icon: 'ℹ️' };
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: '80px',
      right: '20px',
      zIndex: 9999,
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      maxWidth: '320px',
      width: '100%',
    }}>
      {notifications.map(n => {
        const style = getTypeStyle(n.type);
        return (
          <div
            key={n.id}
            style={{
              background: 'rgba(10,14,23,0.95)',
              backdropFilter: 'blur(12px)',
              border: `1px solid ${style.borderColor}`,
              borderRadius: '10px',
              padding: '10px 14px',
              boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
              animation: 'slideInRight 0.4s ease-out',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              cursor: 'pointer',
            }}
            onClick={() => removeNotification(n.id)}
          >
            <span style={{ fontSize: '18px' }}>{style.icon}</span>
            <span style={{ flex: 1, fontSize: '13px', color: '#E0E0E0' }}>{n.message}</span>
            <button
              onClick={(e) => { e.stopPropagation(); removeNotification(n.id); }}
              style={{
                background: 'none',
                border: 'none',
                color: '#6B7280',
                fontSize: '14px',
                cursor: 'pointer',
              }}
            >✕</button>
          </div>
        );
      })}
      <style>
        {`
          @keyframes slideInRight {
            from { transform: translateX(100px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
          }
        `}
      </style>
    </div>
  );
};

export default NotificationSystem;