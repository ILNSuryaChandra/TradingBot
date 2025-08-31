import React, { useState, useEffect } from 'react';

const RiskManagement = ({ api }) => {
  const [riskConfig, setRiskConfig] = useState({
    max_daily_loss_percent: 2.0,
    max_position_size_percent: 10.0,
    default_stop_loss_percent: 1.0,
    max_concurrent_positions: 5,
    max_leverage: 1.0,
    circuit_breaker_loss_percent: 5.0
  });

  const [riskSummary, setRiskSummary] = useState({
    total_capital: 10000,
    available_capital: 10000,
    daily_pnl: 0,
    daily_pnl_percent: 0,
    active_positions: 0,
    circuit_breaker_active: false
  });

  const [loading, setLoading] = useState(false);
  const [updateMessage, setUpdateMessage] = useState('');

  useEffect(() => {
    loadRiskSummary();
  }, []);

  const loadRiskSummary = async () => {
    try {
      // In a real implementation, this would fetch from /api/risk/summary
      // For now, we'll use mock data
      setRiskSummary({
        total_capital: 10000,
        available_capital: 9750,
        daily_pnl: 245.50,
        daily_pnl_percent: 0.02455,
        active_positions: 0,
        daily_trades_count: 3,
        circuit_breaker_active: false,
        capital_utilization_percent: 0.025
      });
    } catch (error) {
      console.error('Error loading risk summary:', error);
    }
  };

  const updateRiskConfig = async () => {
    setLoading(true);
    setUpdateMessage('');
    
    try {
      // In a real implementation, this would POST to /api/risk/config
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setUpdateMessage('Risk configuration updated successfully');
      setTimeout(() => setUpdateMessage(''), 3000);
    } catch (error) {
      setUpdateMessage('Failed to update risk configuration');
      console.error('Error updating risk config:', error);
    } finally {
      setLoading(false);
    }
  };

  const resetCircuitBreaker = async () => {
    try {
      // In a real implementation, this would POST to /api/risk/reset-circuit-breaker
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setRiskSummary(prev => ({
        ...prev,
        circuit_breaker_active: false
      }));
      
      setUpdateMessage('Circuit breaker reset successfully');
      setTimeout(() => setUpdateMessage(''), 3000);
    } catch (error) {
      setUpdateMessage('Failed to reset circuit breaker');
      console.error('Error resetting circuit breaker:', error);
    }
  };

  const formatCurrency = (value) => {
    return value?.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }) || '$0.00';
  };

  const formatPercent = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const getRiskLevelColor = (percent, threshold) => {
    if (percent >= threshold * 0.8) return 'text-red-400 bg-red-900/20';
    if (percent >= threshold * 0.6) return 'text-yellow-400 bg-yellow-900/20';
    return 'text-green-400 bg-green-900/20';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white">Risk Management</h1>
        <p className="text-gray-400 mt-1">Monitor and configure trading risk parameters</p>
      </div>

      {/* Risk Summary Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Capital */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Total Capital</p>
              <p className="text-2xl font-bold text-white">
                {formatCurrency(riskSummary.total_capital)}
              </p>
            </div>
            <div className="p-3 bg-blue-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
              </svg>
            </div>
          </div>
        </div>

        {/* Daily P&L */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Daily P&L</p>
              <p className={`text-2xl font-bold ${
                riskSummary.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(riskSummary.daily_pnl)}
              </p>
              <p className={`text-sm ${
                riskSummary.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatPercent(riskSummary.daily_pnl_percent)}
              </p>
            </div>
            <div className={`p-3 rounded-full ${
              riskSummary.daily_pnl >= 0 ? 'bg-green-600' : 'bg-red-600'
            }`}>
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          </div>
        </div>

        {/* Available Capital */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Available Capital</p>
              <p className="text-2xl font-bold text-white">
                {formatCurrency(riskSummary.available_capital)}
              </p>
              <p className="text-sm text-gray-400">
                {formatPercent(riskSummary.available_capital / riskSummary.total_capital)} available
              </p>
            </div>
            <div className="p-3 bg-purple-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Active Positions */}
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Active Positions</p>
              <p className="text-2xl font-bold text-white">
                {riskSummary.active_positions}
              </p>
              <p className="text-sm text-gray-400">
                of {riskConfig.max_concurrent_positions} max
              </p>
            </div>
            <div className="p-3 bg-orange-600 rounded-full">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Alerts */}
      {riskSummary.circuit_breaker_active && (
        <div className="bg-red-900/20 border border-red-500 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0">
                <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-medium text-red-400">Circuit Breaker Active</h3>
                <p className="text-red-300">Trading has been automatically suspended due to risk limits being exceeded.</p>
              </div>
            </div>
            <button
              onClick={resetCircuitBreaker}
              className="btn-danger"
            >
              Reset
            </button>
          </div>
        </div>
      )}

      {/* Risk Limits Status */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Limits Status</h3>
        
        <div className="space-y-4">
          {/* Daily Loss Limit */}
          <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <div className={`w-3 h-3 rounded-full ${
                  Math.abs(riskSummary.daily_pnl_percent) < (riskConfig.max_daily_loss_percent / 100) * 0.8
                    ? 'bg-green-500' : 'bg-yellow-500'
                }`}></div>
              </div>
              <div>
                <p className="text-white font-medium">Daily Loss Limit</p>
                <p className="text-gray-400 text-sm">
                  Current: {formatPercent(Math.abs(riskSummary.daily_pnl_percent))} / 
                  Limit: {formatPercent(riskConfig.max_daily_loss_percent / 100)}
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="w-32 bg-gray-600 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    Math.abs(riskSummary.daily_pnl_percent) / (riskConfig.max_daily_loss_percent / 100) > 0.8
                      ? 'bg-red-500' : 'bg-green-500'
                  }`}
                  style={{
                    width: `${Math.min(100, (Math.abs(riskSummary.daily_pnl_percent) / (riskConfig.max_daily_loss_percent / 100)) * 100)}%`
                  }}
                ></div>
              </div>
            </div>
          </div>

          {/* Position Size Limit */}
          <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <div>
                <p className="text-white font-medium">Position Size Limit</p>
                <p className="text-gray-400 text-sm">
                  Max per position: {formatPercent(riskConfig.max_position_size_percent / 100)}
                </p>
              </div>
            </div>
            <div className="text-green-400 text-sm">
              Within Limits
            </div>
          </div>

          {/* Concurrent Positions */}
          <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
            <div className="flex items-center space-x-4">
              <div className="flex-shrink-0">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <div>
                <p className="text-white font-medium">Concurrent Positions</p>
                <p className="text-gray-400 text-sm">
                  Active: {riskSummary.active_positions} / Max: {riskConfig.max_concurrent_positions}
                </p>
              </div>
            </div>
            <div className="text-right">
              <div className="w-32 bg-gray-600 rounded-full h-2">
                <div 
                  className="h-2 rounded-full bg-blue-500"
                  style={{
                    width: `${(riskSummary.active_positions / riskConfig.max_concurrent_positions) * 100}%`
                  }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Configuration */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Configuration</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Max Daily Loss (%)
            </label>
            <input
              type="number"
              value={riskConfig.max_daily_loss_percent}
              onChange={(e) => setRiskConfig(prev => ({
                ...prev,
                max_daily_loss_percent: parseFloat(e.target.value)
              }))}
              className="form-input w-full"
              min="0.1"
              max="10"
              step="0.1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Max Position Size (%)
            </label>
            <input
              type="number"
              value={riskConfig.max_position_size_percent}
              onChange={(e) => setRiskConfig(prev => ({
                ...prev,
                max_position_size_percent: parseFloat(e.target.value)
              }))}
              className="form-input w-full"
              min="1"
              max="50"
              step="1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Default Stop Loss (%)
            </label>
            <input
              type="number"
              value={riskConfig.default_stop_loss_percent}
              onChange={(e) => setRiskConfig(prev => ({
                ...prev,
                default_stop_loss_percent: parseFloat(e.target.value)
              }))}
              className="form-input w-full"
              min="0.1"
              max="5"
              step="0.1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Max Concurrent Positions
            </label>
            <input
              type="number"
              value={riskConfig.max_concurrent_positions}
              onChange={(e) => setRiskConfig(prev => ({
                ...prev,
                max_concurrent_positions: parseInt(e.target.value)
              }))}
              className="form-input w-full"
              min="1"
              max="10"
              step="1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Max Leverage
            </label>
            <input
              type="number"
              value={riskConfig.max_leverage}
              onChange={(e) => setRiskConfig(prev => ({
                ...prev,
                max_leverage: parseFloat(e.target.value)
              }))}
              className="form-input w-full"
              min="1"
              max="5"
              step="0.1"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Circuit Breaker Loss (%)
            </label>
            <input
              type="number"
              value={riskConfig.circuit_breaker_loss_percent}
              onChange={(e) => setRiskConfig(prev => ({
                ...prev,
                circuit_breaker_loss_percent: parseFloat(e.target.value)
              }))}
              className="form-input w-full"
              min="1"
              max="20"
              step="0.5"
            />
          </div>
        </div>

        <div className="mt-6 flex items-center space-x-4">
          <button
            onClick={updateRiskConfig}
            disabled={loading}
            className={`btn-primary ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {loading ? 'Updating...' : 'Update Configuration'}
          </button>

          {updateMessage && (
            <div className={`px-3 py-1 rounded-full text-sm ${
              updateMessage.includes('success') 
                ? 'bg-green-900/20 text-green-400' 
                : 'bg-red-900/20 text-red-400'
            }`}>
              {updateMessage}
            </div>
          )}
        </div>
      </div>

      {/* Recent Risk Events */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Risk Events</h3>
        
        <div className="space-y-3">
          <div className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <div className="flex-1">
              <p className="text-white text-sm">Daily P&L within acceptable limits</p>
              <p className="text-gray-400 text-xs">2 minutes ago</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg">
            <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
            <div className="flex-1">
              <p className="text-white text-sm">Position size approaching limit for ETH/USDT</p>
              <p className="text-gray-400 text-xs">15 minutes ago</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <div className="flex-1">
              <p className="text-white text-sm">Stop loss executed successfully on SOL/USDT</p>
              <p className="text-gray-400 text-xs">1 hour ago</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskManagement;