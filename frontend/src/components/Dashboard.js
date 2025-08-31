import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = ({ api, systemHealth }) => {
  const [dashboardData, setDashboardData] = useState({
    pairs: ['ETH/USDT', 'SOL/USDT'],
    currentPrices: {},
    indicators: {},
    signals: {},
    performance: {
      daily_pnl: 0,
      total_trades: 0,
      win_rate: 0,
      active_positions: 0
    }
  });
  const [loading, setLoading] = useState(true);
  const [selectedPair, setSelectedPair] = useState('ETH/USDT');
  const [timeframe, setTimeframe] = useState('1h');

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, [selectedPair, timeframe]);

  const loadDashboardData = async () => {
    try {
      // Get available pairs
      const pairsResponse = await api.get('/pairs');
      
      // Get indicators for selected pair
      const indicatorsResponse = await api.get(`/indicators/${selectedPair}?timeframe=${timeframe}`);
      
      // Get historical data for charts
      const historicalResponse = await api.get(`/historical/${selectedPair}?timeframe=${timeframe}&limit=50`);
      
      setDashboardData(prev => ({
        ...prev,
        pairs: pairsResponse.data.pairs,
        indicators: indicatorsResponse.data.indicators,
        historicalData: historicalResponse.data.data
      }));
      
      setLoading(false);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      setLoading(false);
    }
  };

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getSignalBadge = (signal) => {
    const baseClasses = 'px-2 py-1 rounded-full text-xs font-medium';
    switch (signal) {
      case 'bullish': return `${baseClasses} bg-green-900/20 border border-green-500 text-green-400`;
      case 'bearish': return `${baseClasses} bg-red-900/20 border border-red-500 text-red-400`;
      default: return `${baseClasses} bg-gray-900/20 border border-gray-500 text-gray-400`;
    }
  };

  const formatPrice = (price) => {
    if (typeof price !== 'number') return 'N/A';
    return price.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    });
  };

  const formatPercent = (value) => {
    if (typeof value !== 'number') return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  // Chart data preparation
  const priceChartData = {
    labels: dashboardData.historicalData?.slice(-20).map(item => 
      new Date(item.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    ) || [],
    datasets: [
      {
        label: 'Price',
        data: dashboardData.historicalData?.slice(-20).map(item => item.close) || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.1,
      },
      {
        label: 'VWAP',
        data: dashboardData.indicators?.vwap?.vwap_series?.slice(-20) || [],
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.1,
      }
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#9CA3AF'
        }
      },
      title: {
        display: true,
        text: `${selectedPair} Price & VWAP`,
        color: '#F9FAFB'
      },
    },
    scales: {
      x: {
        ticks: { color: '#9CA3AF' },
        grid: { color: '#374151' }
      },
      y: {
        ticks: { color: '#9CA3AF' },
        grid: { color: '#374151' }
      }
    },
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-700 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-800 rounded-lg p-6 h-32"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Trading Dashboard</h1>
          <p className="text-gray-400 mt-1">Real-time market analysis and trading signals</p>
        </div>
        
        <div className="flex space-x-4">
          <select
            value={selectedPair}
            onChange={(e) => setSelectedPair(e.target.value)}
            className="form-select"
          >
            {dashboardData.pairs.map(pair => (
              <option key={pair} value={pair}>{pair}</option>
            ))}
          </select>
          
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="form-select"
          >
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="15m">15m</option>
            <option value="1h">1h</option>
            <option value="4h">4h</option>
            <option value="1d">1d</option>
          </select>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Current Price */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Current Price</p>
              <p className="text-2xl font-bold text-white">
                {dashboardData.historicalData?.length > 0 
                  ? formatPrice(dashboardData.historicalData[dashboardData.historicalData.length - 1].close)
                  : 'Loading...'
                }
              </p>
            </div>
            <div className="p-3 bg-blue-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
              </svg>
            </div>
          </div>
        </div>

        {/* VWAP */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">VWAP</p>
              <p className="text-2xl font-bold text-white">
                {dashboardData.indicators?.vwap?.vwap 
                  ? formatPrice(dashboardData.indicators.vwap.vwap)
                  : 'Loading...'
                }
              </p>
            </div>
            <div className="p-3 bg-yellow-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          </div>
        </div>

        {/* RSI */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">RSI (14)</p>
              <p className="text-2xl font-bold text-white">
                {dashboardData.indicators?.rsi?.rsi 
                  ? dashboardData.indicators.rsi.rsi.toFixed(2)
                  : 'Loading...'
                }
              </p>
              <p className={`text-sm ${getSignalColor(dashboardData.indicators?.rsi?.rsi_signal)}`}>
                {dashboardData.indicators?.rsi?.rsi_signal || 'neutral'}
              </p>
            </div>
            <div className="p-3 bg-purple-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2-2V7a2 2 0 012-2h2a2 2 0 002 2v2a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 00-2 2h-2a2 2 0 00-2 2v6a2 2 0 01-2 2H9z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Overall Signal */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Overall Signal</p>
              <div className="mt-2">
                <span className={getSignalBadge(dashboardData.indicators?.signals?.overall)}>
                  {dashboardData.indicators?.signals?.overall || 'neutral'}
                </span>
              </div>
            </div>
            <div className="p-3 bg-green-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Charts and Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Price Chart */}
        <div className="chart-container">
          <Line data={priceChartData} options={chartOptions} />
        </div>

        {/* Technical Indicators */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Technical Indicators</h3>
          <div className="space-y-4">
            {/* EMA Alignment */}
            <div className="flex justify-between items-center">
              <span className="text-gray-400">EMA Alignment</span>
              <span className={getSignalBadge(dashboardData.indicators?.ema?.ema_alignment)}>
                {dashboardData.indicators?.ema?.ema_alignment || 'neutral'}
              </span>
            </div>

            {/* MACD */}
            <div className="flex justify-between items-center">
              <span className="text-gray-400">MACD Signal</span>
              <span className={getSignalBadge(dashboardData.indicators?.macd?.macd_crossover)}>
                {dashboardData.indicators?.macd?.macd_crossover || 'neutral'}
              </span>
            </div>

            {/* Volume Analysis */}
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Volume Trend</span>
              <span className={getSignalBadge(dashboardData.indicators?.obv?.obv_signal)}>
                {dashboardData.indicators?.obv?.obv_signal || 'neutral'}
              </span>
            </div>

            {/* Bollinger Bands */}
            <div className="flex justify-between items-center">
              <span className="text-gray-400">BB Position</span>
              <span className="text-sm text-gray-300">
                {dashboardData.indicators?.bollinger?.bb_position || 'neutral'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
            <span className="text-gray-300">Data Service</span>
            <span className={`px-2 py-1 rounded-full text-xs ${
              systemHealth?.services?.data_service === 'active' 
                ? 'bg-green-900/20 border border-green-500 text-green-400'
                : 'bg-red-900/20 border border-red-500 text-red-400'
            }`}>
              {systemHealth?.services?.data_service || 'unknown'}
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
            <span className="text-gray-300">Strategy Engine</span>
            <span className={`px-2 py-1 rounded-full text-xs ${
              systemHealth?.services?.strategy_engine === 'active'
                ? 'bg-green-900/20 border border-green-500 text-green-400'
                : 'bg-red-900/20 border border-red-500 text-red-400'
            }`}>
              {systemHealth?.services?.strategy_engine || 'unknown'}
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
            <span className="text-gray-300">Execution Engine</span>
            <span className={`px-2 py-1 rounded-full text-xs ${
              systemHealth?.services?.execution_engine === 'active'
                ? 'bg-green-900/20 border border-green-500 text-green-400'
                : 'bg-red-900/20 border border-red-500 text-red-400'
            }`}>
              {systemHealth?.services?.execution_engine || 'unknown'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;