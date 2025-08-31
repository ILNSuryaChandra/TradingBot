import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import List, Dict, Optional, Tuple
import aiohttp
import json
import time
from pymongo import MongoClient
from pathlib import Path

logger = logging.getLogger(__name__)

class DataService:
    """
    DataService with two modes: historical (for backtests) and live (for trading),
    both yielding standardized OHLCV frames with timestamps
    """
    
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.exchanges = {}
        self.data_cache = {}
        self.storage_path = Path("data")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize exchanges
        self._initialize_exchanges()
        self._initialize_database()
    
    def _initialize_exchanges(self):
        """Initialize CCXT exchanges"""
        try:
            # Binance for historical data (public API)
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': '',  # Not needed for public endpoints
                'secret': '',
                'sandbox': False,
                'rateLimit': 1200,  # milliseconds
                'enableRateLimit': True,
            })
            
            logger.info("Exchanges initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize exchanges: {e}")
    
    def _initialize_database(self):
        """Initialize MongoDB connection"""
        try:
            mongo_url = os.getenv('MONGO_URL', 'mongodb://localhost:27017/trading_bot')
            self.mongo_client = MongoClient(mongo_url)
            self.db = self.mongo_client.get_default_database()
            logger.info("Database connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    async def health_check(self) -> bool:
        """Check if data service is healthy"""
        try:
            if self.mongo_client:
                # Ping database
                self.mongo_client.admin.command('ping')
                
            # Check exchange connectivity
            if 'binance' in self.exchanges:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.exchanges['binance'].fetch_ticker, 'BTC/USDT'
                )
            
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_historical_ohlcv(
        self,
        pair: str,
        timeframe: str = "1h",
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        exchange: str = "binance"
    ) -> List[Dict]:
        """
        Get historical OHLCV data with fallback strategy:
        1. Try target exchange
        2. Fall back to Binance via CCXT
        3. Cache results locally
        """
        
        try:
            # Check cache first
            cache_key = f"{exchange}_{pair}_{timeframe}_{limit}"
            if cache_key in self.data_cache:
                cached_data, cached_time = self.data_cache[cache_key]
                if time.time() - cached_time < 300:  # 5 minute cache
                    return cached_data
            
            # Normalize pair format for CCXT
            normalized_pair = pair.replace('/', '')
            
            # Try to get data from exchange
            exchange_client = self.exchanges.get(exchange, self.exchanges['binance'])
            
            # Calculate timeframe for pagination
            if start_date and end_date:
                start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
                end_ts = int(datetime.fromisoformat(end_date).timestamp() * 1000)
            else:
                end_ts = int(time.time() * 1000)
                start_ts = end_ts - (limit * self._timeframe_to_ms(timeframe))
            
            ohlcv_data = await self._fetch_ohlcv_paginated(
                exchange_client, pair, timeframe, start_ts, end_ts, limit
            )
            
            # Convert to standardized format
            formatted_data = self._format_ohlcv_data(ohlcv_data)
            
            # Cache the results
            self.data_cache[cache_key] = (formatted_data, time.time())
            
            # Store in database for future reference
            await self._store_historical_data(pair, timeframe, formatted_data)
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {pair}: {e}")
            # Try to get from local storage as fallback
            return await self._get_from_storage(pair, timeframe, limit)
    
    async def _fetch_ohlcv_paginated(
        self,
        exchange_client,
        pair: str,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        limit: int
    ) -> List:
        """Fetch OHLCV data with pagination using since/limit loops"""
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts and len(all_data) < limit:
            try:
                # Fetch batch
                batch = await asyncio.get_event_loop().run_in_executor(
                    None,
                    exchange_client.fetch_ohlcv,
                    pair,
                    timeframe,
                    current_ts,
                    min(1000, limit - len(all_data))  # Binance limit
                )
                
                if not batch:
                    break
                
                all_data.extend(batch)
                
                # Update timestamp for next batch
                current_ts = batch[-1][0] + self._timeframe_to_ms(timeframe)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in pagination: {e}")
                break
        
        # Remove duplicates and sort
        unique_data = {}
        for candle in all_data:
            unique_data[candle[0]] = candle
        
        sorted_data = sorted(unique_data.values(), key=lambda x: x[0])
        
        return sorted_data[:limit]
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 60 * 60 * 1000)
    
    def _format_ohlcv_data(self, ohlcv_data: List) -> List[Dict]:
        """Format raw OHLCV data to standardized format"""
        formatted_data = []
        
        for candle in ohlcv_data:
            formatted_data.append({
                'timestamp': candle[0],
                'datetime': datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })
        
        return formatted_data
    
    async def _store_historical_data(self, pair: str, timeframe: str, data: List[Dict]):
        """Store historical data in database"""
        try:
            if self.db is not None:
                collection_name = f"historical_{pair.replace('/', '_')}_{timeframe}"
                collection = self.db[collection_name]
                
                # Upsert data (update if exists, insert if not)
                for candle in data:
                    collection.update_one(
                        {'timestamp': candle['timestamp']},
                        {'$set': candle},
                        upsert=True
                    )
                
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
    
    async def _get_from_storage(self, pair: str, timeframe: str, limit: int) -> List[Dict]:
        """Get data from local storage as fallback"""
        try:
            if self.db is not None:
                collection_name = f"historical_{pair.replace('/', '_')}_{timeframe}"
                collection = self.db[collection_name]
                
                cursor = collection.find().sort('timestamp', -1).limit(limit)
                data = list(cursor)
                
                # Remove MongoDB _id field
                for item in data:
                    item.pop('_id', None)
                
                return data[::-1]  # Reverse to chronological order
            
        except Exception as e:
            logger.error(f"Error getting data from storage: {e}")
        
        return []
    
    async def stream_live_candles(self, pair: str, timeframe: str = "1m"):
        """Stream live candle data (placeholder for WebSocket implementation)"""
        # This would implement WebSocket connections for live data
        # For now, we'll simulate with periodic API calls
        while True:
            try:
                latest_data = await self.get_historical_ohlcv(
                    pair=pair,
                    timeframe=timeframe,
                    limit=1
                )
                
                if latest_data:
                    yield latest_data[0]
                
                # Wait based on timeframe
                wait_time = self._timeframe_to_ms(timeframe) / 1000
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in live stream: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def get_ticker(self, pair: str, exchange: str = "binance") -> Dict:
        """Get current ticker information"""
        try:
            exchange_client = self.exchanges.get(exchange, self.exchanges['binance'])
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, exchange_client.fetch_ticker, pair
            )
            
            return {
                'symbol': ticker['symbol'],
                'last': float(ticker['last']),
                'bid': float(ticker['bid']) if ticker['bid'] else None,
                'ask': float(ticker['ask']) if ticker['ask'] else None,
                'volume': float(ticker['baseVolume']),
                'timestamp': ticker['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {pair}: {e}")
            return {}
    
    def get_supported_pairs(self) -> List[str]:
        """Get list of supported trading pairs"""
        return ["ETH/USDT", "SOL/USDT", "BTC/USDT"]
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes"""
        return ["1m", "5m", "15m", "1h", "4h", "1d"]