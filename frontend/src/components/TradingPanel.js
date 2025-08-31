import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';

const TradingPanel = ({ api }) => {
  const [selectedPair, setSelectedPair] = useState('ETH/USDT');
  const [strategyType, setStrategyType] = useState('vwap_pullback');
  const [signals, setSignals] = useState([]);
  const [activePositions, setActivePositions] = useState([]);
  const [orderHistory, setOrderHistory] = useState([]);
  const [tradingEnabled, setTradingEnabled] = useState(false);
  const [loading, setLoading] = useState(false);

  const [liveData, setLiveData] = useState({
    price: 0,
    indicators: {},
    currentSignal: null
  });

  useEffect(() => {
    // Simulate live data updates
    const interval = setInterval(loadLiveData, 5000);
    loadOrderHistory();
    return () => clearInterval(interval);
  }, [selectedPair]);

  const loadLiveData = async () => {
    try {
      const [indicatorsResponse, historicalResponse] = await Promise.all([
        api.get(`/indicators/${selectedPair}?timeframe=5m&limit=1`),
        api.get(`/historical/${selectedPair}?timeframe=5m&limit=1`)
      ]);

      const currentPrice = historicalResponse.data.data[0]?.close || 0;
      const indicators = indicatorsResponse.data.indicators || {};

      setLiveData({
        price: currentPrice,
        indicators,
        currentSignal: indicators.signals?.overall || 'neutral'
      });

    } catch (error) {
      console.error('Error loading live data:', error);
    }
  };

  const loadOrderHistory = async () => {
    try {
      // This would fetch real order history from the API
      // For now, we'll use mock data
      setOrderHistory([
        {
          id: '1',
          pair: 'ETH/USDT',
          side: 'buy',
          quantity: 0.5,
          price: 2450.50,
          status: 'filled',
          timestamp: '2024-01-15T10:30:00Z'
        }
      ]);
    } catch (error) {
      console.error('Error loading order history:', error);
    }
  };

  const generateSignal = async () => {
    setLoading(true);
    try {
      // This would generate a trading signal
      const response = await api.get(`/indicators/${selectedPair}?timeframe=5m`);
      const indicators = response.data.indicators;
      
      // Mock signal generation based on indicators
      const signal = {
        id: Date.now().toString(),
        pair: selectedPair,
        strategy: strategyType,
        signal: indicators.signals?.overall || 'neutral',
        confidence: Math.random() * 0.8 + 0.2, // 0.2 to 1.0
        entry_price: liveData.price,
        timestamp: new Date().toISOString(),
        reasons: ['VWAP analysis', 'RSI confirmation', 'Volume analysis']
      };

      setSignals(prev => [signal, ...prev.slice(0, 9)]); // Keep last 10 signals
    } catch (error) {
      console.error('Error generating signal:', error);
    } finally {
      setLoading(false);
    }
  };

  const executeSignal = async (signal) => {
    if (!tradingEnabled) {
      alert('Trading is disabled. Enable trading to execute signals.');
      return;
    }

    try {
      // This would execute the signal as a real order
      console.log('Executing signal:', signal);
      alert('Signal execution simulated - would place real order in live mode');
    } catch (error) {
      console.error('Error executing signal:', error);
    }
  };

  const formatPrice = (price) => {
    return price?.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }) || 'N/A';
  };

  const getSignalColor = (signal) => {
    switch (signal) {
      case 'bullish': return 'text-green-400 bg-green-900/20';
      case 'bearish': return 'text-red-400 bg-red-900/20';
      default: return 'text-gray-400 bg-gray-900/20';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.7) return 'text-green-400';
    if (confidence > 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">Live Trading</h1>
          <p className="text-gray-400 mt-1">VWAP-based trading signals and execution</p>
        </div>

        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={tradingEnabled}
              onChange={(e) => setTradingEnabled(e.target.checked)}
              className="rounded border-gray-600 text-blue-600 shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            />
            <span className="text-sm text-gray-300">Enable Trading</span>
          </label>

          {tradingEnabled && (
            <div className="flex items-center space-x-2 text-green-400">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm">Live Trading Active</span>
            </div>
          )}
        </div>
      </div>

      {/* Trading Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Market Data */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Market Data</h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Trading Pair</label>
              <select
                value={selectedPair}
                onChange={(e) => setSelectedPair(e.target.value)}
                className="form-select w-full"
              >
                <option value="ETH/USDT">ETH/USDT</option>
                <option value="SOL/USDT">SOL/USDT</option>
                <option value="BTC/USDT">BTC/USDT</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Strategy</label>
              <select
                value={strategyType}
                onChange={(e) => setStrategyType(e.target.value)}
                className="form-select w-full"
              >
                <option value="vwap_pullback">VWAP Pullback</option>
                <option value="vwap_breakout">VWAP Breakout</option>
              </select>
            </div>

            <div className="pt-4 border-t border-gray-700">
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-400">Current Price</span>
                <span className="text-white font-mono text-lg">{formatPrice(liveData.price)}</span>
              </div>
              
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-400">VWAP</span>
                <span className="text-white font-mono">{formatPrice(liveData.indicators?.vwap?.vwap)}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400">Current Signal</span>
                <span className={`px-2 py-1 rounded-full text-xs ${getSignalColor(liveData.currentSignal)}`}>
                  {liveData.currentSignal}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Signal Generation */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Signal Generation</h3>
          
          <div className="space-y-4">
            <button
              onClick={generateSignal}
              disabled={loading}
              className={`w-full btn-primary ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {loading ? 'Analyzing...' : 'Generate Signal'}
            </button>

            <div className="text-sm text-gray-400">
              <p>Latest signals based on:</p>
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>VWAP position analysis</li>
                <li>RSI confirmation</li>
                <li>EMA alignment</li>
                <li>Volume confirmation</li>
                <li>MACD crossover</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Risk Summary */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Summary</h3>
          
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Available Capital</span>
              <span className="text-white">$10,000</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Daily P&L</span>
              <span className="text-green-400">+$245.50</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Active Positions</span>
              <span className="text-white">0</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Max Daily Loss</span>
              <span className="text-red-400">2.0%</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-400">Win Rate</span>
              <span className="text-blue-400">65.8%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Signals History */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Signals</h3>
        
        {signals.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-400">No signals generated yet. Click "Generate Signal" to start.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {signals.map((signal) => (
              <div key={signal.id} className="border border-gray-700 rounded-lg p-4">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center space-x-3">
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSignalColor(signal.signal)}`}>
                      {signal.signal.toUpperCase()}
                    </span>
                    <span className="text-gray-300">{signal.pair}</span>
                    <span className="text-gray-400 text-sm">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-white font-mono">{formatPrice(signal.entry_price)}</p>
                    <p className={`text-sm ${getConfidenceColor(signal.confidence)}`}>
                      {(signal.confidence * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="flex flex-wrap gap-2">
                    {signal.reasons.map((reason, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300">
                        {reason}
                      </span>
                    ))}
                  </div>
                  
                  <button
                    onClick={() => executeSignal(signal)}
                    disabled={!tradingEnabled || signal.signal === 'neutral'}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      tradingEnabled && signal.signal !== 'neutral'
                        ? 'bg-blue-600 hover:bg-blue-700 text-white'
                        : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    Execute
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Order History */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Order History</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="table-header">Time</th>
                <th className="table-header">Pair</th>
                <th className="table-header">Side</th>
                <th className="table-header">Quantity</th>
                <th className="table-header">Price</th>
                <th className="table-header">Status</th>
              </tr>
            </thead>
            <tbody>
              {orderHistory.length === 0 ? (
                <tr>
                  <td colSpan="6" className="table-cell text-center py-8 text-gray-400">
                    No orders yet
                  </td>
                </tr>
              ) : (
                orderHistory.map((order) => (
                  <tr key={order.id} className="table-row">
                    <td className="table-cell">
                      {new Date(order.timestamp).toLocaleString()}
                    </td>
                    <td className="table-cell">{order.pair}</td>
                    <td className="table-cell">
                      <span className={`px-2 py-1 rounded text-xs ${
                        order.side === 'buy' ? 'bg-green-900/20 text-green-400' : 'bg-red-900/20 text-red-400'
                      }`}>
                        {order.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="table-cell font-mono">{order.quantity}</td>
                    <td className="table-cell font-mono">{formatPrice(order.price)}</td>
                    <td className="table-cell">
                      <span className={`px-2 py-1 rounded text-xs ${
                        order.status === 'filled' ? 'bg-green-900/20 text-green-400' : 'bg-yellow-900/20 text-yellow-400'
                      }`}>
                        {order.status}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default TradingPanel;