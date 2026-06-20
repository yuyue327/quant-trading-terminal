import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';

interface ProTradePanelProps {
  stockName: string;
  currentPrice: number;
  ohlc: any[];
  onPriceChange?: (price: number) => void;
}

type OrderType = 'market' | 'limit' | 'stop';
type OrderSide = 'buy' | 'sell';

interface Position {
  stock: string;
  shares: number;
  avgPrice: number;
  currentPrice: number;
}

interface Order {
  id: string;
  date: string;
  stock: string;
  side: OrderSide;
  type: OrderType;
  price: number;
  shares: number;
  status: 'pending' | 'filled' | 'cancelled';
}

const ProTradePanel: React.FC<ProTradePanelProps> = ({ stockName, currentPrice, ohlc, onPriceChange }) => {
  const { theme } = useApp();
  const isDark = theme === 'dark';

  const [orderType, setOrderType] = useState<OrderType>('market');
  const [side, setSide] = useState<OrderSide>('buy');
  const [shares, setShares] = useState<number>(100);
  const [limitPrice, setLimitPrice] = useState<number>(0);
  const [stopPrice, setStopPrice] = useState<number>(0);
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [selectedPrice, setSelectedPrice] = useState<number>(0);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [orderBook, setOrderBook] = useState<{ price: number; size: number; side: 'bid' | 'ask' }[]>([]);
  const [flash, setFlash] = useState<'buy' | 'sell' | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  // 从 localStorage 加载数据
  useEffect(() => {
    const saved = localStorage.getItem('pro_positions');
    if (saved) {
      try { setPositions(JSON.parse(saved)); } catch (e) {}
    }
    const savedOrders = localStorage.getItem('pro_orders');
    if (savedOrders) {
      try { setOrders(JSON.parse(savedOrders)); } catch (e) {}
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('pro_positions', JSON.stringify(positions));
  }, [positions]);
  useEffect(() => {
    localStorage.setItem('pro_orders', JSON.stringify(orders));
  }, [orders]);

  // 初始化价格和日期
  useEffect(() => {
    const safePrice = (p: any) => (typeof p === 'number' && !isNaN(p)) ? p : 0;
    const price = safePrice(currentPrice);

    if (ohlc.length > 0) {
      const last = ohlc[ohlc.length - 1];
      const lastPrice = safePrice(last.close);
      if (!selectedDate) {
        setSelectedDate(last.date);
        setSelectedPrice(lastPrice);
        setLimitPrice(lastPrice);
        setStopPrice(lastPrice * 0.95);
        onPriceChange?.(lastPrice);
      } else {
        const found = ohlc.find(d => d.date === selectedDate);
        if (found) {
          const foundPrice = safePrice(found.close);
          setSelectedPrice(foundPrice);
          setLimitPrice(foundPrice);
          setStopPrice(foundPrice * 0.95);
          onPriceChange?.(foundPrice);
        } else {
          setSelectedPrice(lastPrice);
          setLimitPrice(lastPrice);
          setStopPrice(lastPrice * 0.95);
          onPriceChange?.(lastPrice);
        }
      }
    } else {
      setSelectedPrice(price);
      setLimitPrice(price);
      setStopPrice(price * 0.95);
    }
  }, [ohlc, currentPrice, selectedDate, onPriceChange]);

  const handleDateChange = (date: string) => {
    setSelectedDate(date);
    const found = ohlc.find(d => d.date === date);
    if (found) {
      const price = typeof found.close === 'number' ? found.close : 0;
      setSelectedPrice(price);
      setLimitPrice(price);
      setStopPrice(price * 0.95);
      onPriceChange?.(price);
    }
  };

  // 订单簿更新
  useEffect(() => {
    const price = (typeof selectedPrice === 'number' && !isNaN(selectedPrice)) ? selectedPrice : 100;
    const updateOrderBook = () => {
      const bids: { price: number; size: number; side: 'bid' }[] = [];
      const asks: { price: number; size: number; side: 'ask' }[] = [];
      for (let i = 1; i <= 5; i++) {
        const spread = 0.002 * Math.random() + 0.01;
        bids.push({ price: price - i * spread * price, size: Math.floor(Math.random() * 8000 + 2000), side: 'bid' });
        asks.push({ price: price + i * spread * price, size: Math.floor(Math.random() * 8000 + 2000), side: 'ask' });
      }
      setOrderBook([...bids.reverse(), ...asks]);
    };
    updateOrderBook();
    const interval = setInterval(updateOrderBook, 3000);
    return () => clearInterval(interval);
  }, [selectedPrice]);

  const displayPrice = (typeof selectedPrice === 'number' && !isNaN(selectedPrice)) ? selectedPrice : 0;
  const currentPos = positions.find(p => p.stock === stockName);
  const totalPnL = positions.reduce((sum, p) => sum + (displayPrice - p.avgPrice) * p.shares, 0);

  const handleOrder = () => {
    const price = displayPrice;
    const orderPrice = orderType === 'market' ? price : orderType === 'limit' ? limitPrice : stopPrice;
    const order: Order = {
      id: `ORD-${Date.now()}`,
      date: selectedDate || new Date().toISOString().slice(0, 10),
      stock: stockName,
      side,
      type: orderType,
      price: orderPrice,
      shares,
      status: 'filled',
    };
    setOrders([order, ...orders]);
    setFlash(side);
    setToast(`${side === 'buy' ? '🔼 买入' : '🔽 卖出'} ${shares}股 @ ${orderPrice.toFixed(2)} 成功！`);

    if (side === 'buy') {
      const existing = positions.find(p => p.stock === stockName);
      if (existing) {
        const totalCost = existing.avgPrice * existing.shares + orderPrice * shares;
        existing.shares += shares;
        existing.avgPrice = totalCost / existing.shares;
        existing.currentPrice = price;
        setPositions([...positions]);
      } else {
        setPositions([...positions, { stock: stockName, shares, avgPrice: orderPrice, currentPrice: price }]);
      }
    } else {
      const existing = positions.find(p => p.stock === stockName);
      if (existing) {
        const newShares = existing.shares - shares;
        if (newShares <= 0) {
          setPositions(positions.filter(p => p.stock !== stockName));
        } else {
          existing.shares = newShares;
          existing.currentPrice = price;
          setPositions([...positions]);
        }
      }
    }

    setTimeout(() => setToast(null), 3000);
    setTimeout(() => setFlash(null), 500);
  };

  const dates = ohlc.map(d => d.date);

  const panelBg = isDark ? 'rgba(17,24,39,0.95)' : 'rgba(255,255,255,0.95)';
  const panelBorder = isDark ? '#1F2937' : '#E5E7EB';
  const textPrimary = isDark ? '#E0E0E0' : '#111827';
  const textSecondary = isDark ? '#6B7280' : '#4B5563';
  const inputBg = isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
  const inputBorder = isDark ? '#1F2937' : '#D1D5DB';

  return (
    <div style={{
      background: panelBg,
      border: `1px solid ${panelBorder}`,
      borderRadius: '12px',
      padding: '16px',
      height: 'fit-content',
      position: 'relative',
      overflow: 'hidden',
      color: textPrimary,
    }}>
      {flash && (
        <div style={{
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          background: flash === 'buy' ? 'rgba(0,245,160,0.15)' : 'rgba(255,0,80,0.15)',
          animation: 'flashFade 0.6s ease-out',
          pointerEvents: 'none',
        }} />
      )}
      <style>
        {`
          @keyframes flashFade {
            0% { opacity: 1; }
            100% { opacity: 0; }
          }
          @keyframes priceJump {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
          }
          .price-jump { animation: priceJump 0.3s ease; }
        `}
      </style>

      {toast && (
        <div style={{
          position: 'absolute',
          top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          background: isDark ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.95)',
          padding: '12px 24px',
          borderRadius: '8px',
          color: '#00F5A0',
          fontSize: '16px',
          fontWeight: 600,
          zIndex: 100,
          pointerEvents: 'none',
          border: `1px solid ${isDark ? '#00F5A0' : '#00F5A0'}`,
          animation: 'flashFade 2s ease-out',
        }}>
          {toast}
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ color: textPrimary, fontSize: '14px', margin: 0 }}>💼 专业交易面板</h3>
        <span style={{ color: textSecondary, fontSize: '10px' }}>v2.0</span>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', background: inputBg, padding: '8px 12px', borderRadius: '6px' }}>
        <span style={{ color: textSecondary, fontSize: '12px' }}>{stockName}</span>
        <span className={displayPrice !== currentPrice ? 'price-jump' : ''} style={{ color: '#00E5FF', fontWeight: 600, fontSize: '20px' }}>
          ¥{displayPrice.toFixed(2)}
        </span>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <div style={{ color: textSecondary, fontSize: '10px', marginBottom: '2px' }}>📅 交易日期 (回测模式) 价格自动联动</div>
        <select
          value={selectedDate}
          onChange={e => handleDateChange(e.target.value)}
          style={{
            width: '100%',
            background: inputBg,
            border: `1px solid ${inputBorder}`,
            borderRadius: '4px',
            color: textPrimary,
            padding: '4px 8px',
            fontSize: '12px',
          }}
        >
          {dates.slice(-100).map(d => {
            const price = ohlc.find(o => o.date === d)?.close || 0;
            return <option key={d} value={d}>{d} (¥{price.toFixed(2)})</option>;
          })}
        </select>
      </div>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
        <button
          onClick={() => setSide('buy')}
          style={{
            flex: 1,
            background: side === 'buy' ? '#00F5A0' : 'rgba(255,255,255,0.03)',
            color: side === 'buy' ? '#0A0E17' : '#9CA3AF',
            border: 'none',
            borderRadius: '6px',
            padding: '8px',
            fontSize: '14px',
            fontWeight: side === 'buy' ? 600 : 400,
          }}
        >🔼 买入</button>
        <button
          onClick={() => setSide('sell')}
          style={{
            flex: 1,
            background: side === 'sell' ? '#FF0050' : 'rgba(255,255,255,0.03)',
            color: side === 'sell' ? '#FFFFFF' : '#9CA3AF',
            border: 'none',
            borderRadius: '6px',
            padding: '8px',
            fontSize: '14px',
            fontWeight: side === 'sell' ? 600 : 400,
          }}
        >🔽 卖出</button>
      </div>

      <div style={{ display: 'flex', gap: '6px', marginBottom: '10px' }}>
        {(['market', 'limit', 'stop'] as OrderType[]).map(t => (
          <button
            key={t}
            onClick={() => setOrderType(t)}
            style={{
              flex: 1,
              background: orderType === t ? (isDark ? '#374151' : '#D1D5DB') : 'rgba(255,255,255,0.02)',
              color: orderType === t ? (isDark ? '#E0E0E0' : '#111827') : '#6B7280',
              border: `1px solid ${orderType === t ? (isDark ? '#00E5FF' : '#00E5FF') : inputBorder}`,
              borderRadius: '4px',
              padding: '4px 6px',
              fontSize: '10px',
              cursor: 'pointer',
            }}
          >
            {t === 'market' ? '市价' : t === 'limit' ? '限价' : '止损'}
          </button>
        ))}
      </div>

      {orderType !== 'market' && (
        <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
          <div style={{ flex: 1 }}>
            <div style={{ color: textSecondary, fontSize: '10px', marginBottom: '2px' }}>{orderType === 'limit' ? '限价' : '止损价'}</div>
            <input
              type="number"
              value={orderType === 'limit' ? limitPrice : stopPrice}
              onChange={e => {
                const v = parseFloat(e.target.value);
                if (orderType === 'limit') setLimitPrice(v || 0);
                else setStopPrice(v || 0);
              }}
              step={0.01}
              style={{
                width: '100%',
                background: inputBg,
                border: `1px solid ${inputBorder}`,
                borderRadius: '4px',
                color: textPrimary,
                padding: '4px 8px',
                fontSize: '12px',
              }}
            />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ color: textSecondary, fontSize: '10px', marginBottom: '2px' }}>数量</div>
            <input
              type="number"
              value={shares}
              onChange={e => setShares(Math.max(1, parseInt(e.target.value) || 0))}
              step={100}
              min={1}
              style={{
                width: '100%',
                background: inputBg,
                border: `1px solid ${inputBorder}`,
                borderRadius: '4px',
                color: textPrimary,
                padding: '4px 8px',
                fontSize: '12px',
              }}
            />
          </div>
        </div>
      )}
      {orderType === 'market' && (
        <div style={{ marginBottom: '10px' }}>
          <div style={{ color: textSecondary, fontSize: '10px', marginBottom: '2px' }}>数量</div>
          <input
            type="number"
            value={shares}
            onChange={e => setShares(Math.max(1, parseInt(e.target.value) || 0))}
            step={100}
            min={1}
            style={{
              width: '100%',
              background: inputBg,
              border: `1px solid ${inputBorder}`,
              borderRadius: '4px',
              color: textPrimary,
              padding: '4px 8px',
              fontSize: '12px',
            }}
          />
        </div>
      )}

      <button
        onClick={handleOrder}
        style={{
          width: '100%',
          background: side === 'buy' ? '#00F5A0' : '#FF0050',
          color: side === 'buy' ? '#0A0E17' : '#FFFFFF',
          border: 'none',
          borderRadius: '6px',
          padding: '10px',
          fontSize: '14px',
          fontWeight: 600,
          marginBottom: '12px',
          transition: 'transform 0.1s',
        }}
        onMouseDown={e => (e.currentTarget.style.transform = 'scale(0.96)')}
        onMouseUp={e => (e.currentTarget.style.transform = 'scale(1)')}
      >
        {side === 'buy' ? '🔼 买入' : '🔽 卖出'} {shares} 股 @ ¥{(orderType === 'market' ? displayPrice : orderType === 'limit' ? limitPrice : stopPrice).toFixed(2)}
      </button>

      {currentPos && (
        <div style={{
          background: isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.03)',
          borderRadius: '6px',
          padding: '10px',
          marginBottom: '10px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
            <span style={{ color: textSecondary }}>持仓</span>
            <span style={{ color: textPrimary }}>{currentPos.shares} 股</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
            <span style={{ color: textSecondary }}>成本</span>
            <span style={{ color: textPrimary }}>¥{currentPos.avgPrice.toFixed(2)}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px' }}>
            <span style={{ color: textSecondary }}>盈亏</span>
            <span style={{ color: (displayPrice - currentPos.avgPrice) >= 0 ? '#00F5A0' : '#FF0050' }}>
              {((displayPrice - currentPos.avgPrice) / currentPos.avgPrice * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: `1px solid ${inputBorder}`, paddingTop: '10px', marginBottom: '10px' }}>
        <span style={{ color: textSecondary, fontSize: '12px' }}>📊 总盈亏</span>
        <span style={{ color: totalPnL >= 0 ? '#00F5A0' : '#FF0050', fontWeight: 600, fontSize: '15px' }}>
          {totalPnL >= 0 ? '+' : ''}¥{totalPnL.toFixed(2)}
        </span>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <div style={{ color: textSecondary, fontSize: '10px', marginBottom: '4px' }}>📊 订单簿深度 (实时模拟)</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px', fontSize: '10px' }}>
          <div style={{ color: '#FF0050' }}>
            {orderBook.filter(o => o.side === 'ask').slice(0, 5).map((o, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 4px', background: isDark ? 'rgba(255,0,80,0.03)' : 'rgba(255,0,0,0.03)' }}>
                <span>卖 {o.price.toFixed(2)}</span>
                <span>{o.size}</span>
              </div>
            ))}
          </div>
          <div style={{ color: '#00F5A0' }}>
            {orderBook.filter(o => o.side === 'bid').slice(0, 5).map((o, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 4px', background: isDark ? 'rgba(0,245,160,0.03)' : 'rgba(0,255,0,0.03)' }}>
                <span>买 {o.price.toFixed(2)}</span>
                <span>{o.size}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div>
        <div
          onClick={() => setShowHistory(!showHistory)}
          style={{ color: textSecondary, fontSize: '11px', cursor: 'pointer', textDecoration: 'underline' }}
        >
          {showHistory ? '隐藏' : '显示'} 交易记录 ({orders.length})
        </div>
        {showHistory && (
          <div style={{ maxHeight: '150px', overflow: 'auto', marginTop: '6px', fontSize: '11px' }}>
            {orders.slice(0, 20).map(o => (
              <div key={o.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderBottom: `1px solid ${inputBorder}`, color: isDark ? '#9CA3AF' : '#4B5563' }}>
                <span>{o.date}</span>
                <span style={{ color: o.side === 'buy' ? '#00F5A0' : '#FF0050' }}>{o.side === 'buy' ? '买' : '卖'}</span>
                <span>{o.shares}股</span>
                <span>¥{o.price.toFixed(2)}</span>
              </div>
            ))}
            {orders.length === 0 && <span style={{ color: textSecondary }}>暂无交易</span>}
          </div>
        )}
      </div>
    </div>
  );
};

export default ProTradePanel;