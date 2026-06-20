const CACHE_NAME = 'quant-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/src/main.tsx',
  '/src/App.tsx',
  '/src/styles/global.css',
  '/src/components/CandlestickChart.tsx',
  '/src/components/ProTradePanel.tsx',
  '/src/components/PerformanceDashboard.tsx',
  '/src/components/TickerTape.tsx',
  '/src/components/ParticleBackground.tsx',
  '/src/components/AnimatedCounter.tsx',
  '/src/components/ProDataGrid.tsx',
  '/src/components/OrderBookDepth.tsx',
  '/src/components/SystemStatus.tsx',
  '/src/components/MarketSentiment.tsx',
  '/src/components/NotificationSystem.tsx',
  '/src/components/RecentTrades.tsx',
  '/src/components/QuickActions.tsx',
  '/src/components/HeroStats.tsx',
  '/src/components/DonutChart.tsx',
  '/src/components/RiskReturnScatter.tsx',
  '/src/context/AppContext.tsx',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('🔧 Service Worker: 缓存资源中...');
        return cache.addAll(urlsToCache);
      })
      .catch((err) => console.warn('⚠️ 缓存失败:', err))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            console.log('🧹 Service Worker: 清理旧缓存:', name);
            return caches.delete(name);
          }
        })
      );
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        // 如果缓存命中，返回缓存；否则发起网络请求
        return response || fetch(event.request).catch(() => {
          // 如果网络也失败，返回离线页面（可选）
          return new Response('⚠️ 网络已断开，请检查连接', { status: 503 });
        });
      })
  );
});