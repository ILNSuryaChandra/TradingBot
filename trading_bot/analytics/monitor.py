# trading_bot/analytics/monitor.py
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
from dataclasses import dataclass
import json
import asyncio
import telegram
from pathlib import Path

@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_duration: timedelta
    avg_profit_per_trade: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    profit_by_symbol: Dict[str, float]
    hourly_performance: Dict[int, float]

class PerformanceAnalytics:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_thresholds = config['monitoring']['performance_tracking']['adaptive_thresholds']
        self.telegram_bot = self._initialize_telegram() if config.get('telegram_token') else None
        
    def calculate_metrics(self, trades_df: pd.DataFrame) -> PerformanceMetrics:
        try:
            if trades_df.empty:
                return self._get_empty_metrics()
                
            # Basic metrics
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] <= 0])
            
            # Advanced metrics
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum()) / \
                          abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) \
                          if len(trades_df[trades_df['pnl'] <= 0]) > 0 else float('inf')
                          
            # Calculate returns and ratios
            returns = trades_df['pnl'].pct_change()
            total_return = (1 + returns).prod() - 1
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(trades_df)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # Trade statistics
            avg_duration = (trades_df['exit_time'] - trades_df['entry_time']).mean()
            avg_profit = trades_df['pnl'].mean()
            largest_win = trades_df['pnl'].max()
            largest_loss = trades_df['pnl'].min()
            
            # Streak analysis
            consecutive = self._analyze_streaks(trades_df)
            
            # Symbol performance
            profit_by_symbol = trades_df.groupby('symbol')['pnl'].sum().to_dict()
            
            # Hourly performance
            hourly_performance = trades_df.groupby(
                trades_df['entry_time'].dt.hour
            )['pnl'].mean().to_dict()
            
            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                avg_trade_duration=avg_duration,
                avg_profit_per_trade=avg_profit,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive['wins'],
                consecutive_losses=consecutive['losses'],
                profit_by_symbol=profit_by_symbol,
                hourly_performance=hourly_performance
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
            
    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        cumulative_returns = (1 + trades_df['pnl'].pct_change()).cumprod()
        rolling_max = cumulative_returns.expanding(min_periods=1).max()
        drawdowns = cumulative_returns / rolling_max - 1
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
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
        
    def _analyze_streaks(self, trades_df: pd.DataFrame) -> Dict[str, int]:
        if trades_df.empty:
            return {'wins': 0, 'losses': 0}
            
        # Create series of wins/losses
        wins = (trades_df['pnl'] > 0).astype(int)
        
        # Calculate streaks
        win_streaks = (wins != wins.shift()).cumsum()
        max_win_streak = wins.groupby(win_streaks).sum().max()
        max_loss_streak = (~wins.astype(bool)).groupby(win_streaks).sum().max()
        
        return {
            'wins': int(max_win_streak),
            'losses': int(max_loss_streak)
        }
        
    def generate_report(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        return {
            'summary': {
                'total_return_pct': f"{metrics.total_return * 100:.2f}%",
                'win_rate': f"{metrics.win_rate * 100:.2f}%",
                'profit_factor': f"{metrics.profit_factor:.2f}",
                'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}"
            },
            'risk_metrics': {
                'max_drawdown_pct': f"{metrics.max_drawdown * 100:.2f}%",
                'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
                'largest_win': f"${metrics.largest_win:.2f}",
                'largest_loss': f"${metrics.largest_loss:.2f}"
            },
            'trade_statistics': {
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades,
                'avg_profit': f"${metrics.avg_profit_per_trade:.2f}",
                'avg_duration': str(metrics.avg_trade_duration)
            },
            'symbol_performance': {
                symbol: f"${pnl:.2f}"
                for symbol, pnl in metrics.profit_by_symbol.items()
            },
            'hourly_performance': {
                f"{hour:02d}:00": f"${pnl:.2f}"
                for hour, pnl in metrics.hourly_performance.items()
            }
        }
        
    def generate_plots(self, trades_df: pd.DataFrame) -> Dict[str, go.Figure]:
        plots = {}
        
        # Equity curve
        equity_curve = (1 + trades_df['pnl'].pct_change()).cumprod()
        plots['equity_curve'] = go.Figure(
            data=[go.Scatter(x=equity_curve.index, y=equity_curve.values)],
            layout=go.Layout(title='Equity Curve')
        )
        
        # Drawdown chart
        drawdown = self._calculate_drawdown_series(trades_df)
        plots['drawdown'] = go.Figure(
            data=[go.Scatter(x=drawdown.index, y=drawdown.values)],
            layout=go.Layout(title='Drawdown')
        )
        
        # Symbol performance
        symbol_pnl = trades_df.groupby('symbol')['pnl'].sum()
        plots['symbol_performance'] = go.Figure(
            data=[go.Bar(x=symbol_pnl.index, y=symbol_pnl.values)],
            layout=go.Layout(title='Performance by Symbol')
        )
        
        return plots
        
    def _calculate_drawdown_series(self, trades_df: pd.DataFrame) -> pd.Series:
        equity_curve = (1 + trades_df['pnl'].pct_change()).cumprod()
        rolling_max = equity_curve.expanding(min_periods=1).max()
        drawdown = equity_curve / rolling_max - 1
        return drawdown
        
    async def monitor_performance(self, trades_df: pd.DataFrame):
        try:
            # Calculate current metrics
            current_metrics = self.calculate_metrics(trades_df)
            self.metrics_history.append(current_metrics)
            
            # Check for alerts
            alerts = self._check_alerts(current_metrics)
            
            # Send notifications if needed
            if alerts and self.telegram_bot:
                await self._send_alerts(alerts)
                
            # Save metrics history
            self._save_metrics_history()
            
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {str(e)}")
            raise
            
    def _check_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        alerts = []
        
        # Drawdown alert
        if metrics.max_drawdown > self.alert_thresholds['base_thresholds']['drawdown']:
            alerts.append(
                f"High drawdown alert: {metrics.max_drawdown:.2%}"
            )
            
        # Consecutive losses alert
        if metrics.consecutive_losses > self.alert_thresholds['base_thresholds']['losing_streak']:
            alerts.append(
                f"Consecutive losses alert: {metrics.consecutive_losses} trades"
            )
            
        # Performance deterioration alert
        if len(self.metrics_history) > 1:
            prev_metrics = self.metrics_history[-2]
            if metrics.win_rate < prev_metrics.win_rate * 0.8:  # 20% deterioration
                alerts.append(
                    f"Win rate deterioration: {metrics.win_rate:.2%} (prev: {prev_metrics.win_rate:.2%})"
                )
                
        return alerts
        
    async def _send_alerts(self, alerts: List[str]):
        if not self.telegram_bot:
            return
            
        message = "🚨 Trading Bot Alerts:\n\n" + "\n".join(alerts)
        try:
            await self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message
            )
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {str(e)}")
            
    def _initialize_telegram(self) -> Optional[telegram.Bot]:
        try:
            return telegram.Bot(token=self.config['telegram_token'])
        except Exception as e:
            self.logger.error(f"Error initializing Telegram bot: {str(e)}")
            return None
            
    def _save_metrics_history(self):
        try:
            history_path = Path(self.config['monitoring']['metrics_history_path'])
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert metrics history to serializable format
            history_data = [
                {k: str(v) if isinstance(v, timedelta) else v
                 for k, v in metric.__dict__.items()}
                for metric in self.metrics_history
            ]
            
            with open(history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving metrics history: {str(e)}")
            
    def _get_empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            avg_trade_duration=timedelta(0),
            avg_profit_per_trade=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            profit_by_symbol={},
            hourly_performance={}
        )

class AlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.telegram_bot = self._initialize_telegram() if config.get('telegram_token') else None
        
    async def send_alert(self, alert_type: str, message: str):
        try:
            if self.telegram_bot:
                formatted_message = self._format_alert_message(alert_type, message)
                await self._send_telegram_alert(formatted_message)
                
            self._log_alert(alert_type, message)
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            
    def _format_alert_message(self, alert_type: str, message: str) -> str:
        icons = {
            'error': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }
        
        return f"{icons.get(alert_type, '•')} {alert_type.upper()}\n{message}"
        
    async def _send_telegram_alert(self, message: str):
        if not self.telegram_bot:
            return
            
        try:
            await self.telegram_bot.send_message(
                chat_id=self.config['telegram_chat_id'],
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {str(e)}")
            
    def _log_alert(self, alert_type: str, message: str):
        log_path = Path(self.config['monitoring']['alert_log_path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp} - {alert_type.upper()}: {message}\n"
        
        with open(log_path, 'a') as f:
            f.write(log_entry)
