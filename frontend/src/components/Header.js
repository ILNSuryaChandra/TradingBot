import React from 'react';

const Header = ({ systemHealth }) => {
  const formatTime = () => {
    return new Date().toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const getHealthStatus = () => {
    if (!systemHealth) return { status: 'unknown', color: 'gray' };
    
    if (systemHealth.status === 'healthy') {
      return { status: 'Healthy', color: 'green' };
    } else if (systemHealth.status === 'degraded') {
      return { status: 'Degraded', color: 'yellow' };
    } else {
      return { status: 'Unhealthy', color: 'red' };
    }
  };

  const health = getHealthStatus();

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left side - Current time and status */}
        <div className="flex items-center space-x-6">
          <div className="text-sm text-gray-300">
            <span className="font-mono">{formatTime()}</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full bg-${health.color}-500`}></div>
            <span className="text-sm text-gray-300">System {health.status}</span>
          </div>
        </div>

        {/* Right side - System metrics */}
        <div className="flex items-center space-x-6">
          {systemHealth && (
            <>
              {/* Database Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemHealth.services?.database === 'connected' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm text-gray-300">Database</span>
              </div>

              {/* Trading Engine Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemHealth.services?.execution_engine === 'active' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm text-gray-300">Engine</span>
              </div>

              {/* Data Service Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemHealth.services?.data_service === 'active' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm text-gray-300">Data</span>
              </div>
            </>
          )}

          {/* Settings Button */}
          <button className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;