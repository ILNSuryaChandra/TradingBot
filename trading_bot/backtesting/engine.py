# trading_bot/backtesting/engine.py
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path

@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    status: str = 'open'
    exit_reason: Optional[str] = None

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    parameter_set: Dict[str, Any]

class BacktestEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.initial_balance = 10000  # Default starting balance
        self.current_balance = self.initial_balance
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        parameters: Dict[str, Any]
    ) -> BacktestResult:
        try:
            self.trades = []
            self.equity_curve = [self.initial_balance]
            self.current_balance = self.initial_balance
            
            for i in range(len(data)):
                current_data = data.iloc[:i+1]
                if i < 100:  # Skip initial bars to allow for indicator calculation
                    continue
                    
                # Get strategy signal
                signal = strategy(current_data, parameters)
                
                # Process open positions
                self._process_open_positions(current_data.iloc[-1])
                
                # Execute new trades based on signal
                if signal['action'] in ['buy', 'sell']:
                    self._execute_trade(
                        signal,
                        current_data.iloc[-1],
                        parameters
                    )
                    
                # Update equity curve
                self.equity_curve.append(self._calculate_current_equity(current_data.iloc[-1]))
                
            return self._generate_backtest_result(parameters)
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            raise
            
    def _process_open_positions(self, current_bar: pd.Series):
        for trade in [t for t in self.trades if t.status == 'open']:
            # Check stop loss
            if trade.side == 'buy' and current_bar['low'] <= trade.stop_loss:
                self._close_trade(trade, trade.stop_loss, current_bar['timestamp'], 'stop_loss')
            elif trade.side == 'sell' and current_bar['high'] >= trade.stop_loss:
                self._close_trade(trade, trade.stop_loss, current_bar['timestamp'], 'stop_loss')
                
            # Check take profit
            elif trade.side == 'buy' and current_bar['high'] >= trade.take_profit:
                self._close_trade(trade, trade.take_profit, current_bar['timestamp'], 'take_profit')
            elif trade.side == 'sell' and current_bar['low'] <= trade.take_profit:
                self._close_trade(trade, trade.take_profit, current_bar['timestamp'], 'take_profit')
                
    def _execute_trade(
        self,
        signal: Dict[str, Any],
        current_bar: pd.Series,
        parameters: Dict[str, Any]
    ):
        # Calculate position size
        position_size = self._calculate_position_size(
            current_bar['close'],
            parameters['risk_per_trade']
        )
        
        # Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(
            signal['side'],
            current_bar,
            parameters
        )
        
        take_profit = self._calculate_take_profit(
            signal['side'],
            current_bar['close'],
            stop_loss,
            parameters
        )
        
        # Create trade
        trade = BacktestTrade(
            entry_time=current_bar['timestamp'],
            exit_time=None,
            symbol=current_bar.name,
            side=signal['side'],
            entry_price=current_bar['close'],
            exit_price=None,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.trades.append(trade)
        
    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ):
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = 'closed'
        trade.exit_reason = exit_reason
        
        # Calculate PnL
        if trade.side == 'buy':
            trade.pnl = (exit_price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.size
            
        self.current_balance += trade.pnl
        
    def _calculate_position_size(
        self,
        price: float,
        risk_per_trade: float
    ) -> float:
        risk_amount = self.current_balance * risk_per_trade
        return risk_amount / price
        
    def _calculate_stop_loss(
        self,
        side: str,
        current_bar: pd.Series,
        parameters: Dict[str, Any]
    ) -> float:
        atr = current_bar.get('ATR', current_bar['close'] * 0.02)  # Default to 2% if ATR not available
        
        if side == 'buy':
            return current_bar['close'] - (atr * parameters['sl_multiplier'])
        else:
            return current_bar['close'] + (atr * parameters['sl_multiplier'])
            
    def _calculate_take_profit(
        self,
        side: str,
        entry_price: float,
        stop_loss: float,
        parameters: Dict[str, Any]
    ) -> float:
        risk = abs(entry_price - stop_loss)
        
        if side == 'buy':
            return entry_price + (risk * parameters['tp_multiplier'])
        else:
            return entry_price - (risk * parameters['tp_multiplier'])
            
    def _calculate_current_equity(self, current_bar: pd.Series) -> float:
        equity = self.current_balance
        
        # Add unrealized PnL from open positions
        for trade in [t for t in self.trades if t.status == 'open']:
            if trade.side == 'buy':
                equity += (current_bar['close'] - trade.entry_price) * trade.size
            else:
                equity += (trade.entry_price - current_bar['close']) * trade.size
                
        return equity
        
    def _generate_backtest_result(self, parameters: Dict[str, Any]) -> BacktestResult:
        closed_trades = [t for t in self.trades if t.status == 'closed']
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        equity_curve = pd.Series(self.equity_curve)
        returns = equity_curve.pct_change().dropna()
        
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        return BacktestResult(
            total_trades=len(closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(closed_trades) if closed_trades else 0,
            profit_factor=abs(sum(t.pnl for t in winning_trades)) / abs(sum(t.pnl for t in losing_trades))
            if losing_trades else float('inf'),
            total_return=(self.current_balance - self.initial_balance) / self.initial_balance,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            trades=closed_trades,
            equity_curve=equity_curve,
            parameter_set=parameters
        )
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        rolling_max = equity_curve.expanding(min_periods=1).max()
        drawdowns = equity_curve / rolling_max - 1
        return abs(drawdowns.min())
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        return np.sqrt(252) * returns.mean() / downside_std if downside_std != 0 else 0

class WalkForwardOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def optimize(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        parameter_ranges: Dict[str, List[Any]],
        train_size: float = 0.6,
        window_size: int = 1000,
        step_size: int = 500
    ) -> Dict[str, Any]:
        try:
            results = []
            
            # Generate windows
            total_bars = len(data)
            for start_idx in range(0, total_bars - window_size, step_size):
                end_idx = start_idx + window_size
                window_data = data.iloc[start_idx:end_idx]
                
                # Split into training and testing periods
                train_bars = int(window_size * train_size)
                train_data = window_data.iloc[:train_bars]
                test_data = window_data.iloc[train_bars:]
                
                # Optimize parameters on training data
                best_params = self._optimize_window(
                    train_data,
                    strategy,
                    parameter_ranges
                )
                
                # Test parameters on test data
                backtest_engine = BacktestEngine(self.config)
                test_result = backtest_engine.run_backtest(
                    test_data,
                    strategy,
                    best_params
                )
                
                results.append({
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'parameters': best_params,
                    'train_sharpe': test_result.sharpe_ratio,
                    'train_return': test_result.total_return
                })
                
            return self._aggregate_results(results)
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward optimization: {str(e)}")
            raise
            
    def _optimize_window(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        parameter_ranges: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        try:
            # Generate parameter combinations
            param_combinations = [dict(zip(parameter_ranges.keys(), v))
                               for v in product(*parameter_ranges.values())]
                               
            results = []
            backtest_engine = BacktestEngine(self.config)
            
            # Test each parameter combination
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        backtest_engine.run_backtest,
                        data,
                        strategy,
                        params
                    )
                    for params in param_combinations
                ]
                
                for future in futures:
                    result = future.result()
                    results.append({
                        'parameters': result.parameter_set,
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_return': result.total_return
                    })
                    
            # Find best parameters
            sorted_results = sorted(
                results,
                key=lambda x: (x['sharpe_ratio'], x['total_return']),
                reverse=True
            )
            
            return sorted_results[0]['parameters']
            
        except Exception as e:
            self.logger.error(f"Error optimizing window: {str(e)}")
            raise
            
    def _aggregate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            # Calculate parameter stability
            parameter_stats = {}
            for param in results[0]['parameters'].keys():
                values = [r['parameters'][param] for r in results]
                parameter_stats[param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values)
                }
                
            # Calculate performance stability
            performance_stats = {
                'sharpe_ratio': {
                    'mean': np.mean([r['train_sharpe'] for r in results]),
                    'std': np.std([r['train_sharpe'] for r in results])
                },
                'total_return': {
                    'mean': np.mean([r['train_return'] for r in results]),
                    'std': np.std([r['train_return'] for r in results])
                }
            }
            
            return {
                'parameter_stats': parameter_stats,
                'performance_stats': performance_stats,
                'recommended_parameters': self._get_recommended_parameters(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating results: {str(e)}")
            raise
            
    def _get_recommended_parameters(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            # Weight recent results more heavily
            weights = np.linspace(0.5, 1.0, len(results))
            weighted_params = {}
            
            for param in results[0]['parameters'].keys():
                values = [r['parameters'][param] for r in results]
                weighted_average = np.average(values, weights=weights)
                weighted_params[param] = weighted_average
                
            return weighted_params
            
        except Exception as e:
            self.logger.error(f"Error calculating recommended parameters: {str(e)}")
            raise
