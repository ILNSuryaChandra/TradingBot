import React, { useState, useEffect } from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';

const PerformanceAnalytics = ({ api }) => {
  const [analyticsData, setAnalyticsData] = useState({
    performance_summary: {
      total_trades: 0,
      win_rate: 0,
      profit_factor: 0,
      sharpe_ratio: 0,
      max_drawdown: 0,
      total_return: 0
    },
    strategy_performance: [],
    monthly_returns: {},
    trade_distribution: {},
    execution_quality: []
  });

  const [selectedTimeframe, setSelectedTimeframe] = useState('1M');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAnalyticsData();
  }, [selectedTimeframe]);

  const loadAnalyticsData = async () => {
    setLoading(true);
    try {
      // In a real implementation, this would fetch from multiple API endpoints
      // For now, we'll use mock data
      
      const mockData = {
        performance_summary: {
          total_trades: 47,
          win_rate: 0.638,
          profit_factor: 1.85,
          sharpe_ratio: 1.42,
          max_drawdown: 0.086,
          total_return: 0.234,
          avg_trade_duration: '4.2 hours',
          largest_win: 1250.75,
          largest_loss: -680.30
        },
        strategy_performance: [
          { strategy: 'VWAP Pullback', trades: 28, win_rate: 0.714, pnl: 1450.30, sharpe: 1.58 },
          { strategy: 'VWAP Breakout', trades: 19, win_rate: 0.526, pnl: 892.45, sharpe: 1.21 }
        ],
        monthly_returns: {
          '2024-01': 0.045,
          '2024-02': 0.032,
          '2024-03': -0.015,
          '2024-04': 0.067,
          '2024-05': 0.023,
          '2024-06': 0.041,
          '2024-07': 0.038,
          '2024-08': -0.008,
          '2024-09': 0.052,
          '2024-10': 0.029,
          '2024-11': 0.044,
          '2024-12': 0.031
        },
        trade_distribution: {
          'Small Win (0-2%)': 15,
          'Medium Win (2-5%)': 12,
          'Large Win (5%+)': 3,
          'Small Loss (0-2%)': 8,
          'Medium Loss (2-5%)': 7,
          'Large Loss (5%+)': 2
        },
        execution_quality: [
          { date: '2024-12-01', vwap_diff: -0.05, slippage: 0.02 },
          { date: '2024-12-02', vwap_diff: 0.03, slippage: 0.01 },
          { date: '2024-12-03', vwap_diff: -0.02, slippage: 0.03 },
          { date: '2024-12-04', vwap_diff: 0.01, slippage: 0.02 },
          { date: '2024-12-05', vwap_diff: -0.04, slippage: 0.01 }
        ]
      };

      setAnalyticsData(mockData);
    } catch (error) {
      console.error('Error loading analytics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatCurrency = (value) => {
    return value?.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }) || '$0.00';
  };

  // Chart configurations
  const monthlyReturnsChart = {
    labels: Object.keys(analyticsData.monthly_returns || {}),
    datasets: [
      {
        label: 'Monthly Returns',
        data: Object.values(analyticsData.monthly_returns || {}).map(r => r * 100),
        backgroundColor: Object.values(analyticsData.monthly_returns || {}).map(r => 
          r > 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
        ),
        borderColor: Object.values(analyticsData.monthly_returns || {}).map(r => 
          r > 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)'
        ),
        borderWidth: 1,
      }
    ],
  };

  const tradeDistributionChart = {
    labels: Object.keys(analyticsData.trade_distribution || {}),
    datasets: [
      {
        data: Object.values(analyticsData.trade_distribution || {}),
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',   // Small Win
          'rgba(16, 185, 129, 0.8)',  // Medium Win
          'rgba(5, 150, 105, 0.8)',   // Large Win
          'rgba(251, 191, 36, 0.8)',  // Small Loss
          'rgba(245, 158, 11, 0.8)',  // Medium Loss
          'rgba(239, 68, 68, 0.8)',   // Large Loss
        ],
        borderWidth: 1,
      }
    ],
  };

  const executionQualityChart = {
    labels: analyticsData.execution_quality?.map(item => 
      new Date(item.date).toLocaleDateString()
    ) || [],
    datasets: [
      {
        label: 'VWAP Difference (%)',
        data: analyticsData.execution_quality?.map(item => item.vwap_diff) || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        yAxisID: 'y',
      },
      {
        label: 'Slippage (%)',
        data: analyticsData.execution_quality?.map(item => item.slippage) || [],
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        yAxisID: 'y1',
      }
    ],
  };

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
        type: 'linear',
        display: true,
        position: 'left',
        ticks: { color: '#9CA3AF' },
        grid: { color: '#374151' }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        ticks: { color: '#9CA3AF' },
        grid: { drawOnChartArea: false, color: '#374151' }
      },
    },
  };

  const doughnutOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right',
        labels: { color: '#9CA3AF' }
      },
    },
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-700 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
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
          <h1 className="text-3xl font-bold text-white">Performance Analytics</h1>
          <p className="text-gray-400 mt-1">Comprehensive trading performance analysis</p>
        </div>
        
        <div className="flex space-x-2">
          {['1W', '1M', '3M', '6M', '1Y'].map((period) => (
            <button
              key={period}
              onClick={() => setSelectedTimeframe(period)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                selectedTimeframe === period
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {period}
            </button>
          ))}
        </div>
      </div>

      {/* Key Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Total Return</p>
              <p className={`text-2xl font-bold ${
                analyticsData.performance_summary.total_return > 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatPercent(analyticsData.performance_summary.total_return)}
              </p>
            </div>
            <div className="p-3 bg-green-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Win Rate</p>
              <p className="text-2xl font-bold text-white">
                {formatPercent(analyticsData.performance_summary.win_rate)}
              </p>
            </div>
            <div className="p-3 bg-blue-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Sharpe Ratio</p>
              <p className="text-2xl font-bold text-white">
                {analyticsData.performance_summary.sharpe_ratio?.toFixed(2)}
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
                {formatPercent(analyticsData.performance_summary.max_drawdown)}
              </p>
            </div>
            <div className="p-3 bg-red-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Strategy Performance Comparison */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Strategy Performance</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="table-header">Strategy</th>
                <th className="table-header">Trades</th>
                <th className="table-header">Win Rate</th>
                <th className="table-header">P&L</th>
                <th className="table-header">Sharpe Ratio</th>
              </tr>
            </thead>
            <tbody>
              {analyticsData.strategy_performance?.map((strategy, index) => (
                <tr key={index} className="table-row">
                  <td className="table-cell font-medium">{strategy.strategy}</td>
                  <td className="table-cell">{strategy.trades}</td>
                  <td className="table-cell">{formatPercent(strategy.win_rate)}</td>
                  <td className={`table-cell font-mono ${
                    strategy.pnl > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatCurrency(strategy.pnl)}
                  </td>
                  <td className="table-cell">{strategy.sharpe?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Returns */}
        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">Monthly Returns</h3>
          <Bar data={monthlyReturnsChart} options={chartOptions} />
        </div>

        {/* Trade Distribution */}
        <div className="chart-container">
          <h3 className="text-lg font-semibold text-white mb-4">Trade Distribution</h3>
          <Doughnut data={tradeDistributionChart} options={doughnutOptions} />
        </div>
      </div>

      {/* Execution Quality */}
      <div className="chart-container">
        <h3 className="text-lg font-semibold text-white mb-4">Execution Quality</h3>
        <Line data={executionQualityChart} options={chartOptions} />
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Trade Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Trades</span>
              <span className="text-white">{analyticsData.performance_summary.total_trades}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Profit Factor</span>
              <span className="text-white">{analyticsData.performance_summary.profit_factor?.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Trade Duration</span>
              <span className="text-white">{analyticsData.performance_summary.avg_trade_duration}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Largest Win</span>
              <span className="text-green-400">{formatCurrency(analyticsData.performance_summary.largest_win)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Largest Loss</span>
              <span className="text-red-400">{formatCurrency(analyticsData.performance_summary.largest_loss)}</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Volatility (Ann.)</span>
              <span className="text-white">18.5%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Calmar Ratio</span>
              <span className="text-white">2.72</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Sortino Ratio</span>
              <span className="text-white">2.14</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Beta</span>
              <span className="text-white">0.85</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Alpha</span>
              <span className="text-white">12.4%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceAnalytics;