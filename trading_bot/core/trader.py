# trading_bot/core/trader.py
from typing import Dict, List, Optional, Union, Any
import asyncio
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import asdict
import yaml

from .base import AsyncBybitClient, TradePosition, OrderData
from .state_manager import StateManager, TradingState
from .risk_management import RiskManager
from ..models.ml_models import ModelEnsemble, TradingStrategy
from ..backtesting.engine import BacktestEngine, WalkForwardOptimizer

class AutonomousTrader:
    def __init__(self, config: Dict[str, Any]):
        """Initialize with config dict instead of path"""
        self.logger = self._setup_logger()
        self.logger.info("Initializing AutonomousTrader...")
        
        try:
            self.config = config
            self.logger.info("Config loaded successfully")
            
            # Initialize state manager
            self.state_manager = StateManager(self.config)
            self.logger.info("State manager initialized")
            
            # Initialize components
            self.client = AsyncBybitClient(self.config)
            self.logger.info("Client initialized")
            
            self.risk_manager = RiskManager(self.config, self.client)
            self.logger.info("Risk manager initialized")
            
            self.model_ensemble = ModelEnsemble(self.config)
            self.logger.info("Model ensemble initialized")
            
            self.strategy = TradingStrategy(self.config, self.model_ensemble)
            self.logger.info("Strategy initialized")
            
            # Trading state
            self.active_positions: Dict[str, TradePosition] = {}
            self.pending_orders: Dict[str, OrderData] = {}
            self.market_data: Dict[str, pd.DataFrame] = {}
            self.is_running = False
            self.last_model_update = datetime.now()
            self.last_trade_time: Dict[str, datetime] = {}
            
            # Restore state if available
            self._restore_state()
            self.logger.info("AutonomousTrader initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during AutonomousTrader initialization: {str(e)}")
            raise
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {str(e)}")
            
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:  # Only add handlers if none exist
            # File handler
            fh = logging.FileHandler('trading_bot.log')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
        
    async def start(self):
        """Start the trading bot"""
        try:
            self.logger.info("Starting autonomous trader...")
            self.is_running = True
            
            # Initial setup
            await self._initialize_market_data()
            await self._load_models()
            
            # Create and start all trading loops
            loops = [
                self._main_trading_loop(),
                self._market_data_update_loop(),
                self._position_management_loop(),
                self._model_update_loop(),
                self._state_saving_loop()
            ]
            
            # Run all loops concurrently
            await asyncio.gather(*loops)
            
        except Exception as e:
            self.logger.error(f"Error starting trader: {str(e)}")
            self.is_running = False
            raise
            
    async def stop(self):
        """Stop the trading bot gracefully"""
        try:
            self.logger.info("Stopping autonomous trader...")
            self.is_running = False
            
            # Save final state
            self._save_current_state()
            
            # Close all positions first
            try:
                await self._close_all_positions()
            except Exception as e:
                self.logger.error(f"Error closing positions: {str(e)}")
            
            # Then try to cancel all orders
            for symbol in self.config['trading']['symbols']:
                try:
                    await self.client.cancel_all_orders(symbol)
                except Exception as e:
                    self.logger.error(f"Error cancelling orders for {symbol}: {str(e)}")
            
            self.logger.info("Trader stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping trader: {str(e)}")
            # Don't re-raise the exception to allow for graceful shutdown
            
    async def _initialize_market_data(self):
        """Initialize market data for all configured symbols and timeframes"""
        try:
            self.logger.info("Initializing market data...")
            self.market_data = {}
            
            for symbol in self.config['trading']['symbols']:
                self.market_data[symbol] = {}
                for timeframe in (
                    self.config['trading']['timeframes']['lower'] +
                    self.config['trading']['timeframes']['medium'] +
                    self.config['trading']['timeframes']['higher']
                ):
                    try:
                        data = await self.client.get_market_data(
                            symbol=symbol,
                            interval=timeframe,
                            limit=1000
                        )
                        if not data.empty:
                            # Add technical indicators
                            data.ta.strategy(name="AllStrategy")
                            self.market_data[symbol][timeframe] = data
                            self.logger.info(f"Initialized market data for {symbol} {timeframe}")
                        else:
                            self.logger.warning(f"No data received for {symbol} {timeframe}")
                    except Exception as e:
                        self.logger.error(f"Error fetching data for {symbol} {timeframe}: {str(e)}")
                        # Continue with other timeframes even if one fails
                        continue
                        
                    # Small delay between requests to avoid rate limiting
                    await asyncio.sleep(0.5)
                
            if not self.market_data:
                raise Exception("Failed to initialize market data for any symbol")
                
            self.logger.info("Market data initialization completed")
            
        except Exception as e:
            self.logger.error(f"Error initializing market data: {str(e)}")
            raise

    async def _load_models(self):
        """Load or train trading models"""
        try:
            self.logger.info("Loading models...")
            model_path = Path(self.config['models'].get('save_path', 'models'))
            
            if model_path.exists():
                # Try to load existing models
                try:
                    self.model_ensemble.load_models(str(model_path))
                    self.logger.info("Models loaded successfully")
                    return
                except Exception as e:
                    self.logger.warning(f"Could not load existing models: {str(e)}")
            
            # Train new models if loading fails or no models exist
            self.logger.info("Training new models...")
            await self._train_models()
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

    async def _train_models(self):
        """Train new models with available market data"""
        try:
            # Prepare training data
            training_data = {}
            for symbol in self.config['trading']['symbols']:
                if symbol in self.market_data:
                    training_data[symbol] = self._prepare_training_data(symbol)
            
            if not training_data:
                raise Exception("No training data available")
                
            # Train models for each symbol
            for symbol, data in training_data.items():
                self.logger.info(f"Training models for {symbol}")
                self.model_ensemble.train(data)
                
            # Save trained models
            model_path = Path(self.config['models'].get('save_path', 'models'))
            model_path.mkdir(parents=True, exist_ok=True)
            self.model_ensemble.save_models(str(model_path))
            
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def _prepare_training_data(self, symbol: str) -> pd.DataFrame:
        """Prepare training data by combining different timeframes"""
        try:
            # Start with the lowest timeframe data
            base_tf = self.config['trading']['timeframes']['lower'][0]
            if base_tf not in self.market_data[symbol]:
                raise Exception(f"Base timeframe {base_tf} data not available")
                
            data = self.market_data[symbol][base_tf].copy()
            
            # Add features from other timeframes
            for timeframe in (
                self.config['trading']['timeframes']['lower'][1:] +
                self.config['trading']['timeframes']['medium'] +
                self.config['trading']['timeframes']['higher']
            ):
                if timeframe in self.market_data[symbol]:
                    tf_data = self.market_data[symbol][timeframe].copy()
                    
                    # Add timeframe-specific suffix to avoid column name conflicts
                    tf_data = tf_data.add_suffix(f'_{timeframe}')
                    
                    # Merge on timestamp
                    data = pd.merge_asof(
                        data,
                        tf_data,
                        left_index=True,
                        right_index=True,
                        direction='backward'
                    )
            
            return data.dropna()  # Remove any rows with missing data
            
        except Exception as e:
            self.logger.error(f"Error preparing training data for {symbol}: {str(e)}")
            raise
        
    async def _main_trading_loop(self):
        while self.is_running:
            try:
                for symbol in self.config['trading']['symbols']:
                    await self._process_symbol(symbol)
                    
                await asyncio.sleep(
                    self.config['trading']['interval_seconds']
                )
                
            except Exception as e:
                self.logger.error(f"Error in main trading loop: {str(e)}")
                await asyncio.sleep(5)  # Error cooldown
                
    async def _process_symbol(self, symbol: str):
        try:
            # Update market data
            latest_data = self._get_latest_market_data(symbol)
            
            # Get trading signals
            signals = self.strategy.analyze_market(latest_data)
            
            # Check if we should enter a new position
            if signals['signal']['action'] == 'enter':
                await self._enter_position(symbol, signals)
                
            # Check if we should exit any positions
            elif signals['signal']['action'] == 'exit':
                await self._exit_position(symbol, signals)
                
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {str(e)}")
            raise
            
    async def _enter_position(self, symbol: str, signals: Dict[str, Any]):
        try:
            if symbol in self.active_positions:
                return  # Already in position
                
            # Calculate position size
            entry_price = float(self.market_data[symbol]['1']['close'].iloc[-1])
            position_size = await self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=signals['stop_loss']
            )
            
            # Create and execute order
            order = OrderData(
                symbol=symbol,
                side=signals['signal']['side'],
                order_type="Limit",
                qty=position_size,
                price=entry_price,
                stop_loss=signals['stop_loss'],
                take_profit=signals['take_profit']
            )
            
            order_result = await self.client.place_order(order)
            
            if order_result['order_id']:
                self.active_positions[symbol] = TradePosition(
                    symbol=symbol,
                    side=signals['signal']['side'],
                    entry_price=entry_price,
                    size=position_size,
                    stop_loss=signals['stop_loss'],
                    take_profit=signals['take_profit'],
                    entry_time=datetime.now(),
                    position_id=order_result['order_id'],
                    status='open'
                )
                
                self.logger.info(f"Entered new position for {symbol}: {order_result}")
                
        except Exception as e:
            self.logger.error(f"Error entering position for {symbol}: {str(e)}")
            raise
            
    async def _exit_position(self, symbol: str, signals: Dict[str, Any]):
        try:
            if symbol not in self.active_positions:
                return  # No position to exit
                
            position = self.active_positions[symbol]
            
            # Close position
            result = await self.client.close_position(
                symbol=symbol,
                side=position.side
            )
            
            if result.get('order_id'):
                position.status = 'closed'
                del self.active_positions[symbol]
                
                self.logger.info(f"Exited position for {symbol}: {result}")
                
        except Exception as e:
            self.logger.error(f"Error exiting position for {symbol}: {str(e)}")
            raise
            
    async def _market_data_update_loop(self):
        while self.is_running:
            try:
                for symbol in self.config['trading']['symbols']:
                    for timeframe in self.market_data[symbol]:
                        latest_data = await self.client.get_market_data(
                            symbol=symbol,
                            interval=timeframe,
                            limit=100  # Get last 100 candles
                        )
                        
                        self.market_data[symbol][timeframe] = latest_data
                        
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error updating market data: {str(e)}")
                await asyncio.sleep(5)
                
    async def _position_management_loop(self):
        while self.is_running:
            try:
                for symbol, position in list(self.active_positions.items()):
                    # Get current market price
                    current_price = float(
                        self.market_data[symbol]['1']['close'].iloc[-1]
                    )
                    
                    # Check stop loss
                    if position.side == 'buy':
                        if current_price <= position.stop_loss:
                            await self._exit_position(symbol, {'signal': {'action': 'exit'}})
                    else:
                        if current_price >= position.stop_loss:
                            await self._exit_position(symbol, {'signal': {'action': 'exit'}})
                            
                    # Check take profit
                    if position.side == 'buy':
                        if current_price >= position.take_profit:
                            await self._exit_position(symbol, {'signal': {'action': 'exit'}})
                    else:
                        if current_price <= position.take_profit:
                            await self._exit_position(symbol, {'signal': {'action': 'exit'}})
                            
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in position management: {str(e)}")
                await asyncio.sleep(5)
                
    async def _model_update_loop(self):
        while self.is_running:
            try:
                current_time = datetime.now()
                hours_since_update = (
                    current_time - self.last_model_update
                ).total_seconds() / 3600
                
                if hours_since_update >= self.config['models']['update_interval_hours']:
                    await self._train_models()
                    self.last_model_update = current_time
                    
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error updating models: {str(e)}")
                await asyncio.sleep(5)
                
    async def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions = list(self.active_positions.values())
            for position in positions:
                try:
                    await self._exit_position(
                        position.symbol,
                        {'signal': {'action': 'exit'}}
                    )
                except Exception as e:
                    self.logger.error(f"Error closing position {position.symbol}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error in _close_all_positions: {str(e)}")
            raise
            
    async def _cancel_all_orders(self):
        try:
            for symbol in self.config['trading']['symbols']:
                await self.client.cancel_all_orders(symbol)
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {str(e)}")
            raise
            
    def _get_latest_market_data(self, symbol: str) -> pd.DataFrame:
        # Combine data from all timeframes
        data = pd.DataFrame()
        for timeframe in self.market_data[symbol]:
            df = self.market_data[symbol][timeframe].copy()
            # Add indicators
            df.ta.strategy("AllStrategy")
            # Add suffix to avoid column name conflicts
            df = df.add_suffix(f'_{timeframe}')
            data = pd.concat([data, df], axis=1)
        return data
    
    def _restore_state(self):
        """Restore previous trading state if available"""
        try:
            saved_state = self.state_manager.load_state()
            if saved_state:
                self.active_positions = saved_state.active_positions
                self.pending_orders = saved_state.pending_orders
                self.last_model_update = saved_state.last_model_update
                self.last_trade_time = saved_state.last_trade_time
                
                # Verify positions with exchange
                asyncio.create_task(self._verify_positions())
                
                self.logger.info("Trading state restored successfully")
        except Exception as e:
            self.logger.error(f"Error restoring trading state: {str(e)}")

    async def _verify_positions(self):
        """Verify restored positions with exchange"""
        try:
            exchange_positions = await self.client.get_positions()
            exchange_position_map = {
                pos['symbol']: pos for pos in exchange_positions
            }
            
            # Verify each restored position
            for symbol, position in list(self.active_positions.items()):
                if symbol not in exchange_position_map:
                    self.logger.warning(f"Position {symbol} not found on exchange, removing from state")
                    del self.active_positions[symbol]
                else:
                    # Update position with exchange data
                    exchange_pos = exchange_position_map[symbol]
                    position.size = float(exchange_pos['size'])
                    position.entry_price = float(exchange_pos['entry_price'])
                    
        except Exception as e:
            self.logger.error(f"Error verifying positions: {str(e)}")
        
    def _save_current_state(self):
        """Save current trading state"""
        try:
            current_state = TradingState(
                active_positions=self.active_positions,
                pending_orders=self.pending_orders,
                last_model_update=self.last_model_update,
                last_trade_time=self.last_trade_time,
                account_state={
                    'balance': self.current_balance if hasattr(self, 'current_balance') else None,
                },
                market_state={
                    'last_update': datetime.now().isoformat(),
                },
                performance_metrics=self._get_performance_metrics()
            )
            
            self.state_manager.save_state(current_state)
            
        except Exception as e:
            self.logger.error(f"Error saving trading state: {str(e)}")
        
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        try:
            # Implementation depends on your performance tracking
            return {
                'total_trades': 0,  # Replace with actual metrics
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {}

    async def _state_saving_loop(self):
        """Periodically save trading state"""
        while self.is_running:
            try:
                self._save_current_state()
                await asyncio.sleep(60)  # Save state every minute
            except Exception as e:
                self.logger.error(f"Error in state saving loop: {str(e)}")
                await asyncio.sleep(5)
    

if __name__ == "__main__":
    import asyncio
    
    async def main():
        trader = AutonomousTrader("config.json")
        await trader.start()
        
    asyncio.run(main())
