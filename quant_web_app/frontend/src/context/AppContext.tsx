import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

interface Stock {
  stock: string;
  sharpe_median: number;
  annual_return_median: number;
  max_drawdown_median: number;
  win_rate_median: number;
}

interface AppContextType {
  stocks: Stock[];
  setStocks: (stocks: Stock[]) => void;
  selected: string;
  setSelected: (stock: string) => void;
  currentPrice: number;
  setCurrentPrice: (price: number) => void;
  ohlc: any[];
  setOhlc: (data: any[]) => void;
  notifications: { id: string; message: string; type: 'info' | 'warning' | 'success' | 'error'; timestamp: number }[];
  addNotification: (message: string, type: 'info' | 'warning' | 'success' | 'error') => void;
  removeNotification: (id: string) => void;
  isConnected: boolean;
  setIsConnected: (status: boolean) => void;
  latency: number;
  setLatency: (latency: number) => void;
  theme: 'dark' | 'light';
  toggleTheme: () => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selected, setSelected] = useState<string>('');
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [ohlc, setOhlc] = useState<any[]>([]);
  const [notifications, setNotifications] = useState<{ id: string; message: string; type: 'info' | 'warning' | 'success' | 'error'; timestamp: number }[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(true);
  const [latency, setLatency] = useState<number>(12);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  const toggleTheme = useCallback(() => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  const addNotification = useCallback((message: string, type: 'info' | 'warning' | 'success' | 'error') => {
    const id = `notif-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
    setNotifications(prev => [...prev, { id, message, type, timestamp: Date.now() }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  return (
    <AppContext.Provider value={{
      stocks,
      setStocks,
      selected,
      setSelected,
      currentPrice,
      setCurrentPrice,
      ohlc,
      setOhlc,
      notifications,
      addNotification,
      removeNotification,
      isConnected,
      setIsConnected,
      latency,
      setLatency,
      theme,
      toggleTheme,
    }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};