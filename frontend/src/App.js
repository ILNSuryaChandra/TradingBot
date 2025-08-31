import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import './App.css';

// Components
import Dashboard from './components/Dashboard';
import TradingPanel from './components/TradingPanel';
import BacktestPanel from './components/BacktestPanel';
import RiskManagement from './components/RiskManagement';
import PerformanceAnalytics from './components/PerformanceAnalytics';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8001/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [systemHealth, setSystemHealth] = useState(null);
  const [currentPage, setCurrentPage] = useState('dashboard');

  // System health check
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await api.get('/health');
        setSystemHealth(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to connect to trading bot API');
        console.error('Health check failed:', err);
      } finally {
        setLoading(false);
      }
    };

    checkSystemHealth();
    
    // Check health every 30 seconds
    const healthInterval = setInterval(checkSystemHealth, 30000);
    
    return () => clearInterval(healthInterval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-300">Initializing Trading Bot...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="bg-red-900/20 border border-red-500 rounded-lg p-6 max-w-md">
            <h2 className="text-red-400 text-lg font-semibold mb-2">System Error</h2>
            <p className="text-gray-300 mb-4">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-white">
        <div className="flex h-screen">
          {/* Sidebar */}
          <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
          
          {/* Main Content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Header */}
            <Header systemHealth={systemHealth} />
            
            {/* Page Content */}
            <main className="flex-1 overflow-y-auto p-6">
              <Routes>
                <Route 
                  path="/" 
                  element={<Navigate to="/dashboard" replace />} 
                />
                <Route 
                  path="/dashboard" 
                  element={<Dashboard api={api} systemHealth={systemHealth} />} 
                />
                <Route 
                  path="/trading" 
                  element={<TradingPanel api={api} />} 
                />
                <Route 
                  path="/backtest" 
                  element={<BacktestPanel api={api} />} 
                />
                <Route 
                  path="/risk" 
                  element={<RiskManagement api={api} />} 
                />
                <Route 
                  path="/analytics" 
                  element={<PerformanceAnalytics api={api} />} 
                />
              </Routes>
            </main>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;