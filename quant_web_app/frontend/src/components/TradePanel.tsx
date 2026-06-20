import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface TradePanelProps {
  stockName: string;
  currentPrice: number;
  onTrade?: (action: 'buy' | 'sell', shares: number, price: number) => void;
}

interface Position {
  stock: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
}

const TradePanel: React.FC<TradePanelProps> = ({ stockName, currentPrice, onTrade }) => {
  const [shares, setShares] = useState<number>(100);
  const [positions, setPositions] = useState<Position[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // 从 localStorage 加载持仓
  useEffect(() => {
    const saved = localStorage.getItem('positions');
    if (saved) {
      try {
        setPositions(JSON.parse(saved));
      } catch (e) { /* ignore */ }
    }
  }, []);

  // 保存持仓到 localStorage
  useEffect(() => {
    localStorage.setItem('positions', JSON.stringify(positions));
  }, [positions]);

  const handleTrade = (action: 'buy' | 'sell') => {
    if (action === 'buy') {
      const newPos = [...positions];
      const existing = newPos.find(p => p.stock === stockName);
      if (existing) {
        existing.shares += shares;
        existing.avgPrice = (existing.avgPrice * existing.shares + currentPrice * shares) / (existing.shares + shares);
      } else {
        newPos.push({ stock: stockName, shares, avgPrice: currentPrice, currentPrice });
      }
      setPositions(newPos);
    } else {
      const newPos = positions
        .map(p => {
          if (p.stock === stockName) {
            const newShares = p.shares - shares;
            return { ...p, shares: Math.max(0, newShares) };
          }
          return p;
        })
        .filter(p => p.shares > 0);
      setPositions(newPos);
    }
    onTrade?.(action, shares, currentPrice);
  };

  const totalPnL = positions.reduce((sum, p) => {
    const pnl = (currentPrice - p.avgPrice) * p.shares;
    return sum + pnl;
  }, 0);

  const currentPos = positions.find(p => p.stock === stockName);

  return (
    <div style={{
      background: 'rgba(17, 24, 39, 0.9)',
      border: '1px solid #1F2937',
      borderRadius: '12px',
      padding: '16px',
      marginTop: '10px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ color: '#E0E0E0', fontSize: '14px', margin: 0 }}>💼 交易面板</h3>
        <span style={{ color: '#6B7280', fontSize: '12px' }}>{stockName}</span>
      </div>

      {/* 当前价格 */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
        <span style={{ color: '#6B7280', fontSize: '13px' }}>当前价格</span>
        <span style={{ color: '#00E5FF', fontWeight: 600, fontSize: '16px' }}>¥{currentPrice.toFixed(2)}</span>
      </div>

      {/* 持仓 */}
      {currentPos && (
        <div style={{
          background: 'rgba(255,255,255,0.03)',
          borderRadius: '8px',
          padding: '10px',
          marginBottom: '12px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#9CA3AF' }}>
            <span>持仓: {currentPos.shares} 股</span>
            <span>成本: ¥{currentPos.avgPrice.toFixed(2)}</span>
            <span style={{ color: (currentPrice - currentPos.avgPrice) >= 0 ? '#00F5A0' : '#FF0050' }}>
              {(currentPrice - currentPos.avgPrice) >= 0 ? '+' : ''}
              {((currentPrice - currentPos.avgPrice) / currentPos.avgPrice * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      )}

      {/* 交易输入 */}
      <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
        <input
          type="number"
          value={shares}
          onChange={e => setShares(Math.max(1, parseInt(e.target.value) || 0))}
          min={1}
          step={100}
          style={{
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid #1F2937',
            borderRadius: '6px',
            color: '#E0E0E0',
            padding: '6px 10px',
            width: '80px',
            fontSize: '13px',
          }}
        />
        <button
          onClick={() => handleTrade('buy')}
          style={{
            background: '#00F5A0',
            color: '#0A0E17',
            border: 'none',
            borderRadius: '6px',
            padding: '6px 16px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            flex: 1,
          }}
        >
          🔼 买入
        </button>
        <button
          onClick={() => handleTrade('sell')}
          style={{
            background: '#FF0050',
            color: '#FFFFFF',
            border: 'none',
            borderRadius: '6px',
            padding: '6px 16px',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            flex: 1,
          }}
        >
          🔽 卖出
        </button>
      </div>

      {/* 总盈亏 */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        borderTop: '1px solid #1F2937',
        paddingTop: '10px',
        marginTop: '4px',
      }}>
        <span style={{ color: '#6B7280', fontSize: '13px' }}>📊 总盈亏</span>
        <span style={{
          color: totalPnL >= 0 ? '#00F5A0' : '#FF0050',
          fontWeight: 600,
          fontSize: '15px',
        }}>
          {totalPnL >= 0 ? '+' : ''}¥{totalPnL.toFixed(2)}
        </span>
      </div>

      {/* 持仓列表 */}
      {positions.length > 0 && (
        <div style={{ marginTop: '10px' }}>
          <div
            onClick={() => setShowHistory(!showHistory)}
            style={{
              color: '#6B7280',
              fontSize: '12px',
              cursor: 'pointer',
              textDecoration: 'underline',
            }}
          >
            {showHistory ? '隐藏' : '显示'}持仓明细
          </div>
          {showHistory && (
            <div style={{ marginTop: '6px', fontSize: '12px' }}>
              {positions.map(p => (
                <div key={p.stock} style={{ display: 'flex', justifyContent: 'space-between', color: '#9CA3AF', padding: '2px 0' }}>
                  <span>{p.stock.slice(0, 12)}</span>
                  <span>{p.shares}股</span>
                  <span style={{ color: (currentPrice - p.avgPrice) >= 0 ? '#00F5A0' : '#FF0050' }}>
                    {((currentPrice - p.avgPrice) / p.avgPrice * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TradePanel;