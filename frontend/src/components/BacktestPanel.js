import React, { useState } from 'react';
import { Line, Bar } from 'react-chartjs-2';

const BacktestPanel = ({ api }) => {
  const [backtestConfig, setBacktestConfig] = useState({
    pair: 'ETH/USDT',
    timeframe: '1h',
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    initial_capital: 10000,
    strategy_type: 'vwap_pullback'
  });

  const [backtestResults, setBacktestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/backtest', backtestConfig);
      setBacktestResults(response.data.backtest_results);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to run backtest');
      console.error('Backtest error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPercent = (value) => {
    if (typeof value !== 'number') return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatCurrency = (value) => {
    if (typeof value !== 'number') return 'N/A';
    return value.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    });
  };

  // Prepare chart data
  const equityChartData = backtestResults?.equity_curve ? {
    labels: backtestResults.equity_curve.map((_, index) => index),
    datasets: [
      {
        label: 'Portfolio Value',
        data: backtestResults.equity_curve.map(point => point.total_value),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.1,
      }
    ],
  } : null;

  const monthlyReturnsData = backtestResults?.monthly_returns ? {
    labels: Object.keys(backtestResults.monthly_returns),
    datasets: [
      {
        label: 'Monthly Returns',
        data: Object.values(backtestResults.monthly_returns).map(r => r * 100),
        backgroundColor: Object.values(backtestResults.monthly_returns).map(r => 
          r > 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
        ),
      }
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#9CA3AF' }
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Strategy Backtesting</h1>
        <p className="text-gray-400 mt-1">Test your VWAP strategies on historical data</p>
      </div>

      {/* Configuration */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Backtest Configuration</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Trading Pair</label>
            <select
              value={backtestConfig.pair}
              onChange={(e) => setBacktestConfig(prev => ({ ...prev, pair: e.target.value }))}
              className="form-select w-full"
            >
              <option value="ETH/USDT">ETH/USDT</option>
              <option value="SOL/USDT">SOL/USDT</option>
              <option value="BTC/USDT">BTC/USDT</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Timeframe</label>
            <select
              value={backtestConfig.timeframe}
              onChange={(e) => setBacktestConfig(prev => ({ ...prev, timeframe: e.target.value }))}
              className="form-select w-full"
            >
              <option value="5m">5 minutes</option>
              <option value="15m">15 minutes</option>
              <option value="1h">1 hour</option>
              <option value="4h">4 hours</option>
              <option value="1d">1 day</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Strategy</label>
            <select
              value={backtestConfig.strategy_type}
              onChange={(e) => setBacktestConfig(prev => ({ ...prev, strategy_type: e.target.value }))}
              className="form-select w-full"
            >
              <option value="vwap_pullback">VWAP Pullback</option>
              <option value="vwap_breakout">VWAP Breakout</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Start Date</label>
            <input
              type="date"
              value={backtestConfig.start_date}
              onChange={(e) => setBacktestConfig(prev => ({ ...prev, start_date: e.target.value }))}
              className="form-input w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">End Date</label>
            <input
              type="date"
              value={backtestConfig.end_date}
              onChange={(e) => setBacktestConfig(prev => ({ ...prev, end_date: e.target.value }))}
              className="form-input w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Initial Capital</label>
            <input
              type="number"
              value={backtestConfig.initial_capital}
              onChange={(e) => setBacktestConfig(prev => ({ ...prev, initial_capital: parseFloat(e.target.value) }))}
              className="form-input w-full"
              min="1000"
              step="1000"
            />
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={runBacktest}
            disabled={loading}
            className={`btn-primary ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {loading ? 'Running Backtest...' : 'Run Backtest'}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-900/20 border border-red-500 rounded-lg">
            <p className="text-red-400">{error}</p>
          </div>
        )}
      </div>

      {/* Results */}
      {backtestResults && (
        <>
          {/* Performance Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400">Total Return</p>
                  <p className={`text-2xl font-bold ${
                    backtestResults.total_return > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatPercent(backtestResults.total_return)}
                  </p>
                </div>
                <div className="p-3 bg-blue-600 rounded-full">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400">Sharpe Ratio</p>
                  <p className="text-2xl font-bold text-white">
                    {backtestResults.sharpe_ratio?.toFixed(2) || 'N/A'}
                  </p>
                </div>
                <div className="p-3 bg-purple-600 rounded-full">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2-2V7a2 2 0 012-2h2a2 2 0 002 2v2a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 00-2 2h-2a2 2 0 00-2 2v6a2 2 0 01-2 2H9z" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400">Max Drawdown</p>
                  <p className="text-2xl font-bold text-red-400">
                    {formatPercent(backtestResults.max_drawdown)}
                  </p>
                </div>
                <div className="p-3 bg-red-600 rounded-full">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400">Win Rate</p>
                  <p className="text-2xl font-bold text-white">
                    {formatPercent(backtestResults.win_rate)}
                  </p>
                </div>
                <div className="p-3 bg-green-600 rounded-full">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
            </div>
          </div>

          {/* Additional Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Trading Statistics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Trades</span>
                  <span className="text-white">{backtestResults.total_trades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Winning Trades</span>
                  <span className="text-green-400">{backtestResults.winning_trades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Losing Trades</span>
                  <span className="text-red-400">{backtestResults.losing_trades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Average Win</span>
                  <span className="text-green-400">{formatCurrency(backtestResults.avg_win)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Average Loss</span>
                  <span className="text-red-400">{formatCurrency(backtestResults.avg_loss)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Profit Factor</span>
                  <span className="text-white">{backtestResults.profit_factor?.toFixed(2) || 'N/A'}</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Return Metrics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Annualized Return</span>
                  <span className={`${backtestResults.annualized_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {formatPercent(backtestResults.annualized_return)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Best Trade</span>
                  <span className="text-green-400">{formatCurrency(backtestResults.largest_win)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Worst Trade</span>
                  <span className="text-red-400">{formatCurrency(backtestResults.largest_loss)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Initial Capital</span>
                  <span className="text-white">{formatCurrency(backtestConfig.initial_capital)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Final Value</span>
                  <span className="text-white">
                    {formatCurrency(backtestConfig.initial_capital * (1 + backtestResults.total_return))}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Equity Curve */}
            {equityChartData && (
              <div className="chart-container">
                <h3 className="text-lg font-semibold text-white mb-4">Equity Curve</h3>
                <Line data={equityChartData} options={chartOptions} />
              </div>
            )}

            {/* Monthly Returns */}
            {monthlyReturnsData && (
              <div className="chart-container">
                <h3 className="text-lg font-semibold text-white mb-4">Monthly Returns</h3>
                <Bar data={monthlyReturnsData} options={chartOptions} />
              </div>
            )}
          </div>

          {/* Trade Details */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Trade Details</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="table-header">Entry Time</th>
                    <th className="table-header">Exit Time</th>
                    <th className="table-header">Side</th>
                    <th className="table-header">Entry Price</th>
                    <th className="table-header">Exit Price</th>
                    <th className="table-header">Quantity</th>
                    <th className="table-header">P&L</th>
                    <th className="table-header">P&L %</th>
                  </tr>
                </thead>
                <tbody>
                  {backtestResults.trades?.slice(0, 10).map((trade, index) => (
                    <tr key={index} className="table-row">
                      <td className="table-cell text-sm">
                        {new Date(trade.entry_time).toLocaleString()}
                      </td>
                      <td className="table-cell text-sm">
                        {trade.exit_time ? new Date(trade.exit_time).toLocaleString() : 'Open'}
                      </td>
                      <td className="table-cell">
                        <span className={`px-2 py-1 rounded text-xs ${
                          trade.side === 'buy' ? 'bg-green-900/20 text-green-400' : 'bg-red-900/20 text-red-400'
                        }`}>
                          {trade.side?.toUpperCase()}
                        </span>
                      </td>
                      <td className="table-cell font-mono">{formatCurrency(trade.entry_price)}</td>
                      <td className="table-cell font-mono">{formatCurrency(trade.exit_price)}</td>
                      <td className="table-cell font-mono">{trade.quantity?.toFixed(4)}</td>
                      <td className={`table-cell font-mono ${trade.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatCurrency(trade.pnl)}
                      </td>
                      <td className={`table-cell font-mono ${trade.pnl_percent > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {formatPercent(trade.pnl_percent)}
                      </td>
                    </tr>
                  )) || (
                    <tr>
                      <td colSpan="8" className="table-cell text-center py-4 text-gray-400">
                        No trades to display
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default BacktestPanel;