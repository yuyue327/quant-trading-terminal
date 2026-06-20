import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './styles/global.css';
import { useApp } from './context/AppContext';
import CandlestickChart from './components/CandlestickChart';
import ProTradePanel from './components/ProTradePanel';
import PerformanceDashboard from './components/PerformanceDashboard';
import TickerTape from './components/TickerTape';
import ParticleBackground from './components/ParticleBackground';
import AnimatedCounter from './components/AnimatedCounter';
import ProDataGrid from './components/ProDataGrid';
import OrderBookDepth from './components/OrderBookDepth';
import SystemStatus from './components/SystemStatus';
import MarketSentiment from './components/MarketSentiment';
import NotificationSystem from './components/NotificationSystem';
import RecentTrades from './components/RecentTrades';
import QuickActions from './components/QuickActions';
import HeroStats from './components/HeroStats';
import DonutChart from './components/DonutChart';
import RiskReturnScatter from './components/RiskReturnScatter';

// 读取后端API基础地址
const API_BASE = import.meta.env.VITE_API_BASE_URL;

interface Stock {
  stock: string;
  sharpe_median: number;
  annual_return_median: number;
  max_drawdown_median: number;
  win_rate_median: number;
  total_trades_median?: number;
}

const stockSectors: Record<string, string> = {
  'A_sh.600036_招商银行': '银行',
  'A_sz.000001_平安银行': '银行',
  'A_sh.600030_中信证券': '券商',
  'A_sz.000858_五粮液': '消费',
  'A_sh.600519_贵州茅台': '消费',
  'A_sz.300750_宁德时代': '新能源',
  'A_sh.600887_伊利股份': '消费',
  'A_sh.601012_隆基绿能': '新能源',
  'A_sh.601688_华泰证券': '券商',
  'A_sz.000333_美的集团': '家电',
  'A_sz.000568_泸州老窖': '消费',
  'A_sz.000651_格力电器': '家电',
  'A_sz.002142_宁波银行': '银行',
  'A_sz.002594_比亚迪': '新能源',
  'A_sz.300059_东方财富': '券商',
  'US_AAPL_AAPL': '科技',
  'US_MSFT_MSFT': '科技',
  'US_NVDA_NVDA': '科技',
};

function App() {
  const {
    stocks,
    setStocks,
    selected,
    setSelected,
    currentPrice,
    setCurrentPrice,
    ohlc,
    setOhlc,
    addNotification,
    theme,
    toggleTheme,
  } = useApp();

  const [loading, setLoading] = useState<boolean>(true);
  const [filter, setFilter] = useState<string>('全部');
  const [showShortcuts, setShowShortcuts] = useState<boolean>(false);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    axios
      .get(`${API_BASE}/api/stocks`)
      .then(res => {
        if (res.data?.stocks) {
          const stockList = res.data.stocks.map((name: string) => ({
            stock: name,
            sharpe_median: 0,
            annual_return_median: 0,
            max_drawdown_median: 0,
            win_rate_median: 0,
          }));
          setStocks(stockList);
          if (stockList.length > 0) setSelected(stockList[0].stock);

          axios
            .get(`${API_BASE}/api/summary`)
            .then(summaryRes => {
              if (summaryRes.data?.data) {
                const summaryData = summaryRes.data.data;
                setStocks(prev => prev.map(s => {
                  const found = summaryData.find((item: any) => item.stock === s.stock);
                  return found ? { ...s, ...found } : s;
                }));
              }
            })
            .catch(() => {});
        }
        setLoading(false);
        setTimeout(() => setIsLoaded(true), 300);
        addNotification('🟢 系统已启动，数据加载完成', 'success');
      })
      .catch(() => {
        setLoading(false);
        setTimeout(() => setIsLoaded(true), 300);
        addNotification('⚠️ 数据加载失败，请检查后端服务', 'error');
      });
  }, [setStocks, setSelected, addNotification]);

  useEffect(() => {
    if (!selected) return;
    axios
      .get(`${API_BASE}/api/ohlc/${selected}`)
      .then(res => {
        const data = res.data.data || [];
        setOhlc(data);
        if (data.length > 0) setCurrentPrice(data[data.length - 1].close);
      })
      .catch(() => {
        addNotification(`❌ 无法加载 ${selected} 的行情数据`, 'error');
      });
  }, [selected, setOhlc, setCurrentPrice, addNotification]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        const idx = stocks.findIndex(s => s.stock === selected);
        if (idx === -1) return;
        const newIdx = e.key === 'ArrowLeft' ? Math.max(0, idx - 1) : Math.min(stocks.length - 1, idx + 1);
        if (newIdx !== idx) setSelected(stocks[newIdx].stock);
      }
      if (e.key >= '1' && e.key <= '9') {
        const idx = parseInt(e.key) - 1;
        if (idx < stocks.length) setSelected(stocks[idx].stock);
      }
      if (e.key >= '1' && e.key <= '9' && e.shiftKey) {
        const idx = parseInt(e.key) - 1;
        const sectors = ['全部', '银行', '券商', '消费', '新能源', '科技', '家电'];
        if (idx < sectors.length) setFilter(sectors[idx]);
      }
      if (e.key === '?' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setShowShortcuts(!showShortcuts);
      }
      if (e.key === 'f' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        document.querySelector<HTMLSelectElement>('select')?.focus();
      }
      if (e.key === 'Escape') setShowShortcuts(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [stocks, selected, setSelected, showShortcuts, filter, setFilter]);

  const sectors = ['全部', ...new Set(Object.values(stockSectors))];
  const filteredStocks =
    filter === '全部' ? stocks : stocks.filter(s => stockSectors[s.stock] === filter);

  if (loading) {
    return (
      <div style={{ background: '#0A0E17', color: '#E0E0E0', height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: '16px' }}>
        <div style={{ width: '40px', height: '40px', border: '3px solid #1F2937', borderTopColor: '#00E5FF', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
        <style>{'@keyframes spin { to { transform: rotate(360deg); } }'}</style>
        <h2>⏳ 加载量化数据中...</h2>
      </div>
    );
  }

  const currentStock = stocks.find(s => s.stock === selected);
  const isDark = theme === 'dark';
  const bg = isDark ? '#0A0E17' : '#F3F4F6';
  const textColor = isDark ? '#E0E0E0' : '#111827';

  return (
    <div
      className="app-container"
      style={{
        background: bg,
        color: textColor,
        minHeight: '100vh',
        padding: '12px 20px',
        transition: 'background 0.3s, color 0.3s',
        position: 'relative',
        zIndex: 1,
        opacity: isLoaded ? 1 : 0,
        animation: isLoaded ? 'fadeInScale 0.8s ease-out' : 'none',
      }}
    >
      <ParticleBackground />
      <NotificationSystem />

      <div
        className="corner-tag"
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          zIndex: 999,
          background: isDark ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.6)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255,255,255,0.03)',
          borderRadius: '8px',
          padding: '6px 14px',
          fontSize: '9px',
          color: isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)',
          letterSpacing: '2px',
          textTransform: 'uppercase',
          userSelect: 'none',
          pointerEvents: 'none',
        }}
      >
        ⚡ QUANT CORE v4.0 · 自适应专家系统
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
        <TickerTape />
        <SystemStatus />
      </div>

      <div className="app-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h1 style={{ fontSize: '22px', fontWeight: 700 }} className="gradient-text">
            📊 自适应量化专家系统
          </h1>
          <span style={{ fontSize: '10px', background: 'rgba(0,229,255,0.15)', color: '#00E5FF', padding: '2px 10px', borderRadius: '12px', border: '1px solid rgba(0,229,255,0.2)' }}>PRO</span>
        </div>
        <div className="app-controls" style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button onClick={toggleTheme} style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid #1F2937', borderRadius: '6px', padding: '6px 10px', color: '#9CA3AF', fontSize: '14px', cursor: 'pointer', transition: 'all 0.3s' }}>
            {isDark ? '☀️' : '🌙'}
          </button>
          <select value={filter} onChange={e => setFilter(e.target.value)} style={{ background: isDark ? 'rgba(255,255,255,0.05)' : '#E5E7EB', color: isDark ? '#E0E0E0' : '#111827', border: '1px solid #1F2937', borderRadius: '6px', padding: '6px 12px', fontSize: '13px', cursor: 'pointer', transition: 'all 0.3s' }}>
            {sectors.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={selected} onChange={e => setSelected(e.target.value)} style={{ background: isDark ? 'rgba(255,255,255,0.05)' : '#E5E7EB', color: isDark ? '#E0E0E0' : '#111827', border: '1px solid #1F2937', borderRadius: '6px', padding: '6px 16px', fontSize: '14px', cursor: 'pointer', minWidth: '180px', transition: 'all 0.3s' }}>
            {filteredStocks.map(s => <option key={s.stock} value={s.stock}>{s.stock} {stockSectors[s.stock] ? `(${stockSectors[s.stock]})` : ''}</option>)}
          </select>
          <span style={{ color: '#6B7280', fontSize: '11px', background: 'rgba(255,255,255,0.03)', padding: '4px 8px', borderRadius: '4px' }}>← → · 1-9 · ⌘F · ⌘?</span>
          {showShortcuts && (
            <div style={{ position: 'absolute', top: '70px', right: '20px', background: isDark ? 'rgba(17,24,39,0.95)' : 'rgba(229,231,235,0.95)', backdropFilter: 'blur(20px)', border: '1px solid #1F2937', borderRadius: '8px', padding: '16px', zIndex: 1000, minWidth: '220px', boxShadow: '0 20px 60px rgba(0,0,0,0.8)', animation: 'fadeInScale 0.3s ease-out' }}>
              <h4 style={{ marginBottom: '8px', color: '#00E5FF' }}>⌨️ 快捷键</h4>
              <div style={{ fontSize: '12px', color: '#9CA3AF', lineHeight: '1.8' }}>
                <div>← → 切换股票</div>
                <div>1-9 快速切换标的</div>
                <div>Shift+1-9 切换行业</div>
                <div>⌘F 聚焦搜索</div>
                <div>⌘? 打开/关闭此面板</div>
                <div>Esc 关闭此面板</div>
              </div>
              <button onClick={() => setShowShortcuts(false)} style={{ marginTop: '10px', background: '#374151', border: 'none', borderRadius: '4px', padding: '4px 12px', color: '#E0E0E0', fontSize: '12px', cursor: 'pointer' }}>关闭</button>
            </div>
          )}
        </div>
      </div>

      <QuickActions />

      {currentStock && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px', marginBottom: '16px' }}>
          <div className={`glass-card metric-card-1 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">夏普比率</div>
            <div className="metric-value cyan">
              <AnimatedCounter value={currentStock.sharpe_median || 0} duration={1200} decimals={3} />
            </div>
          </div>
          <div className={`glass-card metric-card-2 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">年化收益</div>
            <div className="metric-value green">
              <AnimatedCounter value={(currentStock.annual_return_median || 0) * 100} duration={1200} decimals={2} suffix="%" />
            </div>
          </div>
          <div className={`glass-card metric-card-3 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">最大回撤</div>
            <div className="metric-value red">
              <AnimatedCounter value={(currentStock.max_drawdown_median || 0) * 100} duration={1200} decimals={2} suffix="%" />
            </div>
          </div>
          <div className={`glass-card metric-card-4 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">胜率</div>
            <div className="metric-value gold">
              <AnimatedCounter value={(currentStock.win_rate_median || 0) * 100} duration={1200} decimals={1} suffix="%" />
            </div>
          </div>
          <div className={`glass-card metric-card-5 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">股票池</div>
            <div className="metric-value gold">{stocks.length}</div>
          </div>
          <div className={`glass-card metric-card-6 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">当前价格</div>
            <div className="metric-value cyan">¥<AnimatedCounter value={currentPrice} duration={800} decimals={2} /></div>
          </div>
          <div className={`glass-card metric-card-7 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">数据天数</div>
            <div className="metric-value">{ohlc.length}</div>
          </div>
          <div className={`glass-card metric-card-8 animate-fadeInUp ${isDark ? '' : 'glass-card-light'}`}>
            <div className="metric-label">最新价</div>
            <div className="metric-value" style={{ color: currentPrice > 0 ? '#00F5A0' : '#6B7280' }}>
              {currentPrice > 0 ? '🟢 活跃' : '⏸ 停牌'}
            </div>
          </div>
        </div>
      )}

      <HeroStats />

      <div style={{ height: '1px', background: `linear-gradient(90deg, transparent, ${isDark ? 'rgba(0,229,255,0.15)' : 'rgba(0,229,255,0.3)'}, transparent)`, margin: '20px 0' }} />

      <div className="charts-row" style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '16px', marginBottom: '20px' }}>
        <DonutChart />
        <RiskReturnScatter />
      </div>

      <div style={{ height: '1px', background: `linear-gradient(90deg, transparent, ${isDark ? 'rgba(0,229,255,0.15)' : 'rgba(0,229,255,0.3)'}, transparent)`, margin: '20px 0' }} />

      <div className="main-layout" style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        <div style={{ flex: '1', minWidth: '300px' }}>
          <ProDataGrid
            title="📊 策略绩效排名"
            subtitle="按夏普比率排序 · 实时数据"
            data={stocks.map(s => ({
              stock: s.stock,
              sharpe: s.sharpe_median || 0,
              annual_return: s.annual_return_median || 0,
              max_drawdown: s.max_drawdown_median || 0,
              win_rate: s.win_rate_median || 0,
              sector: stockSectors[s.stock] || '其他',
            }))}
            columns={[
              { key: 'stock', label: '标的', type: 'string', sortable: false },
              { key: 'sector', label: '行业', type: 'string', sortable: true },
              { key: 'sharpe', label: '夏普比率', type: 'number', sortable: true, format: (v: number) => v.toFixed(3) },
              { key: 'annual_return', label: '年化收益', type: 'percent', sortable: true },
              { key: 'max_drawdown', label: '最大回撤', type: 'percent', sortable: true },
              { key: 'win_rate', label: '胜率', type: 'percent', sortable: true },
            ]}
            onRowClick={row => {
              const stock = stocks.find(s => s.stock === row.stock);
              if (stock) setSelected(stock.stock);
            }}
          />
        </div>
        <div style={{ width: '280px', minWidth: '200px' }}>
          <MarketSentiment />
          <RecentTrades />
        </div>
      </div>

      <div className="main-layout" style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginTop: '16px' }}>
        <div style={{ flex: '1', minWidth: '300px' }}>
          <CandlestickChart />
        </div>
        <div className="trade-panel-wrapper" style={{ width: '300px', minWidth: '240px' }}>
          <ProTradePanel
            stockName={selected}
            currentPrice={currentPrice}
            ohlc={ohlc}
            onPriceChange={price => setCurrentPrice(price)}
          />
        </div>
      </div>

      <div style={{ marginTop: '16px' }}>
        <OrderBookDepth currentPrice={currentPrice} stockName={selected} />
      </div>

      <div style={{ marginTop: '20px' }}>
        <PerformanceDashboard ohlc={ohlc} probs={[]} stockName={selected} />
      </div>

      <div style={{ marginTop: '20px', color: '#6B7280', fontSize: '11px', textAlign: 'center', borderTop: `1px solid ${isDark ? '#1F2937' : '#E5E7EB'}`, paddingTop: '12px' }}>
        ⚡ 模拟交易演示 | 数据来源: A股 + 美股 2021-2026 | 仅供研究参考，不构成投资建议
        <span style={{ marginLeft: '16px' }}>键盘 ← → 切换股票 · 1-9 快速跳转 · ⌘? 查看快捷键</span>
      </div>
    </div>
  );
}

export default App;