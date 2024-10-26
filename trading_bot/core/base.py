# trading_bot/core/base.py
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import yaml
import numpy as np
import pandas as pd
import pandas_ta as ta
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Core Data Structures
@dataclass
class TradePosition:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    position_id: str
    status: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
@dataclass
class OrderData:
    symbol: str
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    time_in_force: str = "GTC"
    reduce_only: bool = False
    close_on_trigger: bool = False
    
@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Dict[str, float]

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    STABLE = "stable"

class AsyncBybitClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = HTTP(
            testnet=config['api']['testnet'],
            api_key=config['api']['api_key'],
            api_secret=config['api']['api_secret']
        )
        self.rate_limit_margin = config['api']['rate_limit_margin']
        self.logger = logging.getLogger(__name__)
        
    async def get_balance(self, coin: str = "USDT") -> float:
        try:
            response = await self._make_request(
                lambda: self.client.get_wallet_balance(accountType="UNIFIED", coin=coin)
            )
            if response.get('retCode') == 0:
                return float(response['result']['list'][0]['totalWalletBalance'])
            raise Exception(f"Failed to get balance: {response.get('retMsg')}")
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            raise

    async def get_market_data(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        try:
            response = await self._make_request(
                lambda: self.client.get_kline(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
            )
            
            if response.get('retCode') == 0:
                df = pd.DataFrame(response['result']['list'])
                if not df.empty:
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Convert string values to float
                    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                        df[col] = df[col].astype(float)
                    
                    return df.sort_values('timestamp')
                return pd.DataFrame()
            raise Exception(f"Failed to get market data: {response.get('retMsg')}")
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise

    async def place_order(self, order: 'OrderData') -> Dict[str, Any]:
        try:
            response = await self._make_request(
                lambda: self.client.place_order(
                    category="linear",
                    symbol=order.symbol,
                    side=order.side,
                    orderType=order.order_type,
                    qty=str(order.qty),
                    price=str(order.price) if order.price else None,
                    stopLoss=str(order.stop_loss) if order.stop_loss else None,
                    takeProfit=str(order.take_profit) if order.take_profit else None,
                    timeInForce=order.time_in_force,
                    reduceOnly=order.reduce_only,
                    closeOnTrigger=order.close_on_trigger
                )
            )
            
            if response.get('retCode') == 0:
                return response['result']
            raise Exception(f"Failed to place order: {response.get('retMsg')}")
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            response = await self._make_request(
                lambda: self.client.get_positions(
                    category="linear",
                    symbol=symbol
                )
            )
            
            if response.get('retCode') == 0:
                return response['result']['list']
            raise Exception(f"Failed to get positions: {response.get('retMsg')}")
        except Exception as e:
            self.logger.error(f"Error fetching positions: {str(e)}")
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        try:
            response = await self._make_request(
                lambda: self.client.cancel_order(
                    category="linear",
                    symbol=symbol,
                    orderId=order_id
                )
            )
            
            if response.get('retCode') == 0:
                return response['result']
            raise Exception(f"Failed to cancel order: {response.get('retMsg')}")
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            raise

    async def close_position(self, symbol: str, side: str) -> Dict[str, Any]:
        try:
            # Get position size first
            positions = await self.get_positions(symbol)
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if position:
                response = await self._make_request(
                    lambda: self.client.place_order(
                        category="linear",
                        symbol=symbol,
                        side="Sell" if side == "Buy" else "Buy",
                        orderType="Market",
                        qty=position['size'],
                        reduceOnly=True
                    )
                )
                
                if response.get('retCode') == 0:
                    return response['result']
                raise Exception(f"Failed to close position: {response.get('retMsg')}")
            return {"message": "No position to close"}
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise

    async def _make_request(self, request_func, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(self.rate_limit_margin)  # Rate limiting
                return request_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            raise
            
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()
        
    def update_config(self, new_config: Dict[str, Any]):
        try:
            self.config.update(new_config)
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file)
        except Exception as e:
            logging.error(f"Error updating config: {str(e)}")
            raise

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.trades_df = pd.DataFrame()
        self.positions_df = pd.DataFrame()
        self._initialize_database()
        
    def _initialize_database(self):
        try:
            # Load existing data or create new DataFrames
            try:
                self.trades_df = pd.read_parquet(f"{self.db_path}/trades.parquet")
                self.positions_df = pd.read_parquet(f"{self.db_path}/positions.parquet")
            except FileNotFoundError:
                self.trades_df = pd.DataFrame(columns=[
                    'timestamp', 'symbol', 'side', 'entry_price', 'exit_price',
                    'size', 'pnl', 'strategy', 'position_id'
                ])
                self.positions_df = pd.DataFrame(columns=[
                    'symbol', 'side', 'entry_price', 'size', 'stop_loss',
                    'take_profit', 'entry_time', 'position_id', 'status'
                ])
        except Exception as e:
            logging.error(f"Error initializing database: {str(e)}")
            raise
            
    def save_trade(self, trade: Dict[str, Any]):
        try:
            self.trades_df = pd.concat([
                self.trades_df,
                pd.DataFrame([trade])
            ])
            self.trades_df.to_parquet(f"{self.db_path}/trades.parquet")
        except Exception as e:
            logging.error(f"Error saving trade: {str(e)}")
            raise
            
    def save_position(self, position: TradePosition):
        try:
            self.positions_df = pd.concat([
                self.positions_df,
                pd.DataFrame([position.__dict__])
            ])
            self.positions_df.to_parquet(f"{self.db_path}/positions.parquet")
        except Exception as e:
            logging.error(f"Error saving position: {str(e)}")
            raise
            
    def get_trades_history(self) -> pd.DataFrame:
        return self.trades_df.copy()
        
    def get_open_positions(self) -> pd.DataFrame:
        return self.positions_df[self.positions_df['status'] == 'open'].copy()
