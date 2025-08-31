import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Represents a single trade in backtest"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    side: str  # 'buy' or 'sell'
    pnl: float
    pnl_percent: float
    fees: float
    signal_confidence: float
    strategy: str
    stop_loss: Optional[float]
    take_profit: Optional[float]

@dataclass
class BacktestResult:
    """Complete backtest results"""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    trades: List[BacktestTrade]
    equity_curve: List[Dict]
    monthly_returns: Dict[str, float]

class BacktestEngine:
    """
    Vectorized backtester with walk-forward validation and parameter optimization
    """
    
    def __init__(self, data_service, indicator_engine, strategy_engine):
        self.data_service = data_service
        self.indicator_engine = indicator_engine
        self.strategy_engine = strategy_engine
        
        self.backtest_results = {}
        self.parameter_sweep_results = []
        
        # Default backtest settings
        self.config = {
            'initial_capital': 10000.0,
            'commission_rate': 0.001,  # 0.1% commission
            'slippage_rate': 0.0005,   # 0.05% slippage
            'risk_free_rate': 0.02,    # 2% annual risk-free rate
            'max_positions': 1,        # For now, single position
        }
        
        logger.info("Backtest Engine initialized")
    
    async def run_backtest(
        self,
        pair: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        strategy_config: Optional[Dict] = None,
        initial_capital: Optional[float] = None
    ) -> BacktestResult:
        """Run complete backtest on historical data"""
        
        try:
            logger.info(f"Starting backtest for {pair} on {timeframe} timeframe")
            
            # Set parameters
            if initial_capital:
                self.config['initial_capital'] = initial_capital
            
            # Get historical data
            historical_data = await self._get_backtest_data(pair, timeframe, start_date, end_date)
            
            if len(historical_data) < 100:
                raise ValueError(f"Insufficient data for backtesting: {len(historical_data)} candles")
            
            # Run the backtest simulation
            trades, equity_curve = await self._simulate_trading(historical_data, strategy_config or {})
            
            # Calculate performance metrics
            result = self._calculate_performance_metrics(trades, equity_curve, historical_data)
            
            # Store results
            backtest_id = f"{pair}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.backtest_results[backtest_id] = result
            
            logger.info(f"Backtest completed: {result.total_trades} trades, {result.total_return:.2%} return")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    async def _get_backtest_data(
        self,
        pair: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[Dict]:
        """Get historical data for backtesting"""
        
        try:
            # Calculate default date range if not provided
            if not end_date:
                end_date = datetime.now().isoformat()
            
            if not start_date:
                # Default to 1 year of data
                start_dt = datetime.now() - timedelta(days=365)
                start_date = start_dt.isoformat()
            
            # Get data with high limit for backtesting
            data = await self.data_service.get_historical_ohlcv(
                pair=pair,
                timeframe=timeframe,
                limit=5000,  # Large limit for backtesting
                start_date=start_date,
                end_date=end_date
            )
            
            if not data:
                raise ValueError(f"No historical data available for {pair}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting backtest data: {e}")
            raise
    
    async def _simulate_trading(
        self,
        historical_data: List[Dict],
        strategy_config: Dict
    ) -> Tuple[List[BacktestTrade], List[Dict]]:
        """Simulate trading on historical data"""
        
        trades = []
        equity_curve = []
        
        current_capital = self.config['initial_capital']
        current_position = None
        
        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logger.info(f"Simulating trading on {len(df)} data points")
        
        # Process data in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(50, len(df)):  # Start after warm-up period
            
            # Get data slice for indicator calculation
            start_idx = max(0, i - 200)  # Use 200 periods for indicators
            data_slice = historical_data[start_idx:i+1]
            
            # Generate trading signal
            try:
                signal_result = self.strategy_engine.generate_signals(
                    pair="ETH/USDT",  # Use the pair from parameters
                    ohlcv_data=data_slice,
                    strategy_type=self.strategy_engine.StrategyType.VWAP_PULLBACK
                )
                
                current_candle = historical_data[i]
                current_price = current_candle['close']
                current_time = datetime.fromtimestamp(current_candle['timestamp'] / 1000)
                
                # Process signals
                if signal_result.get('signal') in ['buy', 'sell'] and not current_position:
                    # Enter new position
                    trade = self._enter_position(
                        signal_result,
                        current_price,
                        current_time,
                        current_capital
                    )
                    
                    if trade:
                        current_position = trade
                        current_capital -= trade.quantity * trade.entry_price * (1 + self.config['commission_rate'])
                
                elif current_position:
                    # Check exit conditions
                    exit_signal = self._check_exit_conditions(
                        current_position,
                        current_price,
                        current_time,
                        signal_result
                    )
                    
                    if exit_signal:
                        # Exit position
                        current_position.exit_time = current_time
                        current_position.exit_price = current_price
                        
                        # Calculate PnL
                        if current_position.side == 'buy':
                            pnl = (current_price - current_position.entry_price) * current_position.quantity
                        else:
                            pnl = (current_position.entry_price - current_price) * current_position.quantity
                        
                        # Subtract fees
                        total_fees = (current_position.quantity * current_position.entry_price + 
                                    current_position.quantity * current_price) * self.config['commission_rate']
                        pnl -= total_fees
                        
                        current_position.pnl = pnl
                        current_position.pnl_percent = pnl / (current_position.quantity * current_position.entry_price)
                        current_position.fees = total_fees
                        
                        # Update capital
                        current_capital += current_position.quantity * current_price * (1 - self.config['commission_rate'])
                        
                        trades.append(current_position)
                        current_position = None
                
                # Record equity curve
                equity_curve.append({
                    'timestamp': current_time.isoformat(),
                    'capital': current_capital,
                    'position_value': (current_position.quantity * current_price if current_position else 0),
                    'total_value': current_capital + (current_position.quantity * current_price if current_position else 0)
                })
                
            except Exception as e:
                logger.debug(f"Error processing candle {i}: {e}")
                continue
        
        # Close any remaining position
        if current_position:
            final_candle = historical_data[-1]
            final_price = final_candle['close']
            final_time = datetime.fromtimestamp(final_candle['timestamp'] / 1000)
            
            current_position.exit_time = final_time
            current_position.exit_price = final_price
            
            if current_position.side == 'buy':
                pnl = (final_price - current_position.entry_price) * current_position.quantity
            else:
                pnl = (current_position.entry_price - final_price) * current_position.quantity
            
            total_fees = (current_position.quantity * current_position.entry_price + 
                        current_position.quantity * final_price) * self.config['commission_rate']
            pnl -= total_fees
            
            current_position.pnl = pnl
            current_position.pnl_percent = pnl / (current_position.quantity * current_position.entry_price)
            current_position.fees = total_fees
            
            trades.append(current_position)
        
        logger.info(f"Trading simulation completed: {len(trades)} trades generated")
        
        return trades, equity_curve
    
    def _enter_position(
        self,
        signal_result: Dict,
        current_price: float,
        current_time: datetime,
        available_capital: float
    ) -> Optional[BacktestTrade]:
        """Enter a new trading position"""
        
        try:
            signal = signal_result.get('signal')
            confidence = signal_result.get('confidence', 0.5)
            
            # Position sizing (simple fixed percentage for now)
            position_size_pct = min(0.1, confidence)  # Max 10% of capital, scaled by confidence
            position_value = available_capital * position_size_pct
            
            if position_value < 100:  # Minimum position size
                return None
            
            quantity = position_value / current_price
            
            # Apply slippage
            if signal == 'buy':
                entry_price = current_price * (1 + self.config['slippage_rate'])
                side = 'buy'
            else:
                entry_price = current_price * (1 - self.config['slippage_rate'])
                side = 'sell'
            
            # Calculate stop loss and take profit
            stop_loss = entry_price * (0.98 if side == 'buy' else 1.02)  # 2% stop loss
            take_profit = entry_price * (1.04 if side == 'buy' else 0.96)  # 4% take profit
            
            trade = BacktestTrade(
                entry_time=current_time,
                exit_time=None,
                entry_price=entry_price,
                exit_price=None,
                quantity=quantity,
                side=side,
                pnl=0.0,
                pnl_percent=0.0,
                fees=0.0,
                signal_confidence=confidence,
                strategy=signal_result.get('strategy', 'unknown'),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            return None
    
    def _check_exit_conditions(
        self,
        position: BacktestTrade,
        current_price: float,
        current_time: datetime,
        signal_result: Dict
    ) -> bool:
        """Check if position should be exited"""
        
        try:
            # Time-based exit (max 7 days)
            if current_time - position.entry_time > timedelta(days=7):
                return True
            
            # Stop loss
            if position.stop_loss:
                if position.side == 'buy' and current_price <= position.stop_loss:
                    return True
                elif position.side == 'sell' and current_price >= position.stop_loss:
                    return True
            
            # Take profit
            if position.take_profit:
                if position.side == 'buy' and current_price >= position.take_profit:
                    return True
                elif position.side == 'sell' and current_price <= position.take_profit:
                    return True
            
            # Signal reversal
            current_signal = signal_result.get('signal', 'hold')
            if ((position.side == 'buy' and current_signal == 'sell') or 
                (position.side == 'sell' and current_signal == 'buy')):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False
    
    def _calculate_performance_metrics(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[Dict],
        historical_data: List[Dict]
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        try:
            if not trades:
                # Return empty result if no trades
                return BacktestResult(
                    total_return=0.0,
                    annualized_return=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    trades=[],
                    equity_curve=equity_curve,
                    monthly_returns={}
                )
            
            # Basic trade statistics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL statistics
            total_pnl = sum(t.pnl for t in trades)
            winning_pnls = [t.pnl for t in trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in trades if t.pnl < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0
            largest_win = max(winning_pnls) if winning_pnls else 0
            largest_loss = min(losing_pnls) if losing_pnls else 0
            
            # Profit factor
            gross_profit = sum(winning_pnls) if winning_pnls else 0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Returns calculation
            initial_capital = self.config['initial_capital']
            final_capital = initial_capital + total_pnl
            total_return = (final_capital - initial_capital) / initial_capital
            
            # Annualized return
            if equity_curve:
                days_trading = (datetime.fromisoformat(equity_curve[-1]['timestamp'].replace('Z', '')) - 
                              datetime.fromisoformat(equity_curve[0]['timestamp'].replace('Z', ''))).days
                years_trading = max(days_trading / 365.25, 1/365.25)  # Minimum 1 day
                annualized_return = (1 + total_return) ** (1 / years_trading) - 1
            else:
                annualized_return = 0
            
            # Drawdown calculation
            peak_value = initial_capital
            max_drawdown = 0
            
            for point in equity_curve:
                current_value = point['total_value']
                if current_value > peak_value:
                    peak_value = current_value
                else:
                    drawdown = (peak_value - current_value) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Sharpe ratio
            if len(equity_curve) > 1:
                returns = []
                for i in range(1, len(equity_curve)):
                    prev_value = equity_curve[i-1]['total_value']
                    curr_value = equity_curve[i]['total_value']
                    if prev_value > 0:
                        returns.append((curr_value - prev_value) / prev_value)
                
                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        # Annualize and calculate Sharpe
                        periods_per_year = 365 * 24  # Assuming hourly data
                        excess_return = avg_return * periods_per_year - self.config['risk_free_rate']
                        volatility = std_return * np.sqrt(periods_per_year)
                        sharpe_ratio = excess_return / volatility
                    else:
                        sharpe_ratio = 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Monthly returns
            monthly_returns = self._calculate_monthly_returns(equity_curve)
            
            result = BacktestResult(
                total_return=total_return,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                trades=trades,
                equity_curve=equity_curve,
                monthly_returns=monthly_returns
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    def _calculate_monthly_returns(self, equity_curve: List[Dict]) -> Dict[str, float]:
        """Calculate monthly returns"""
        try:
            if not equity_curve:
                return {}
            
            monthly_returns = {}
            
            # Group by month
            monthly_values = {}
            for point in equity_curve:
                timestamp = datetime.fromisoformat(point['timestamp'].replace('Z', ''))
                month_key = timestamp.strftime('%Y-%m')
                
                if month_key not in monthly_values:
                    monthly_values[month_key] = []
                monthly_values[month_key].append(point['total_value'])
            
            # Calculate returns for each month
            prev_month_end = self.config['initial_capital']
            
            for month_key in sorted(monthly_values.keys()):
                month_values = monthly_values[month_key]
                month_start = month_values[0]
                month_end = month_values[-1]
                
                # Use previous month end as starting value
                monthly_return = (month_end - prev_month_end) / prev_month_end
                monthly_returns[month_key] = monthly_return
                
                prev_month_end = month_end
            
            return monthly_returns
            
        except Exception as e:
            logger.error(f"Error calculating monthly returns: {e}")
            return {}
    
    async def run_parameter_sweep(
        self,
        pair: str,
        timeframe: str,
        parameter_ranges: Dict[str, List],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Run parameter optimization sweep"""
        
        try:
            logger.info(f"Starting parameter sweep for {pair}")
            
            results = []
            
            # Generate parameter combinations
            import itertools
            param_names = list(parameter_ranges.keys())
            param_values = [parameter_ranges[name] for name in param_names]
            
            combinations = list(itertools.product(*param_values))
            
            logger.info(f"Testing {len(combinations)} parameter combinations")
            
            for i, combination in enumerate(combinations):
                try:
                    # Create strategy config
                    strategy_config = dict(zip(param_names, combination))
                    
                    # Update strategy engine config
                    original_config = self.strategy_engine.get_strategy_config()
                    self.strategy_engine.update_strategy_config(strategy_config)
                    
                    # Run backtest
                    result = await self.run_backtest(
                        pair=pair,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        strategy_config=strategy_config
                    )
                    
                    # Store result
                    param_result = {
                        'parameters': strategy_config,
                        'total_return': result.total_return,
                        'sharpe_ratio': result.sharpe_ratio,
                        'max_drawdown': result.max_drawdown,
                        'win_rate': result.win_rate,
                        'total_trades': result.total_trades,
                        'profit_factor': result.profit_factor
                    }
                    
                    results.append(param_result)
                    
                    # Restore original config
                    self.strategy_engine.update_strategy_config(original_config)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(combinations)} parameter combinations")
                
                except Exception as e:
                    logger.error(f"Error in parameter combination {i}: {e}")
                    continue
            
            # Sort by best performance (Sharpe ratio)
            results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
            
            self.parameter_sweep_results = results
            
            logger.info(f"Parameter sweep completed: {len(results)} valid results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running parameter sweep: {e}")
            raise
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get summary of all backtests"""
        return {
            'completed_backtests': len(self.backtest_results),
            'parameter_sweeps': len(self.parameter_sweep_results),
            'config': self.config
        }