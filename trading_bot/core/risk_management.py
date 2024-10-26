# trading_bot/core/risk_management.py
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from dataclasses import asdict
from .base import TradePosition, OrderData, AsyncBybitClient

class RiskManager:
    def __init__(self, config: Dict[str, Any], client: AsyncBybitClient):
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.risk_params = config['trading']['position_management']
        self.adaptive_params = config['trading']['adaptive_parameters']
        
    async def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        market_volatility: float = None,
        win_rate: float = None
    ) -> float:
        try:
            account_balance = await self.client.get_balance()
            base_risk = self.risk_params['base_risk_per_trade']
            
            # Adjust risk based on adaptive parameters if enabled
            if self.adaptive_params['enabled']:
                adjusted_risk = self._calculate_adaptive_risk(
                    base_risk,
                    market_volatility,
                    win_rate
                )
            else:
                adjusted_risk = base_risk
                
            risk_amount = account_balance * adjusted_risk
            stop_loss_pct = abs(entry_price - stop_loss) / entry_price
            
            position_size = risk_amount / stop_loss_pct
            return self._apply_position_limits(position_size, account_balance)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            raise
            
    def _calculate_adaptive_risk(
        self,
        base_risk: float,
        market_volatility: Optional[float],
        win_rate: Optional[float]
    ) -> float:
        try:
            risk_multiplier = 1.0
            
            if market_volatility is not None:
                vol_factor = self.adaptive_params['adjustment_factors']['market_volatility']
                risk_multiplier *= (1 - (market_volatility * vol_factor))
                
            if win_rate is not None:
                win_factor = self.adaptive_params['adjustment_factors']['win_rate']
                risk_multiplier *= (1 + ((win_rate - 0.5) * win_factor))
                
            min_risk = self.adaptive_params['position_sizing']['min_risk']
            max_risk = self.adaptive_params['position_sizing']['max_risk']
            
            adjusted_risk = base_risk * risk_multiplier
            return np.clip(adjusted_risk, min_risk, max_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive risk: {str(e)}")
            raise
            
    def _apply_position_limits(self, position_size: float, account_balance: float) -> float:
        max_position_size = account_balance * self.risk_params['max_account_risk']
        return min(position_size, max_position_size)
        
    async def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        market_data: pd.DataFrame
    ) -> float:
        try:
            # Calculate ATR-based stop loss
            atr = market_data.ta.atr(length=14).iloc[-1]
            base_sl_multiplier = self.config['trading']['sl_multiplier']
            
            if self.adaptive_params['enabled']:
                sl_multiplier = self._calculate_adaptive_sl_multiplier(market_data)
            else:
                sl_multiplier = base_sl_multiplier
                
            stop_distance = atr * sl_multiplier
            
            if side == "Buy":
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
                
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            raise
            
    def _calculate_adaptive_sl_multiplier(self, market_data: pd.DataFrame) -> float:
        try:
            base_multiplier = self.config['trading']['adaptive_parameters']['stop_loss']['base_sl_multiplier']
            
            # Calculate volatility
            returns = market_data['close'].pct_change()
            volatility = returns.std()
            
            # Adjust multiplier based on volatility
            vol_weight = self.config['trading']['adaptive_parameters']['stop_loss']['adaptive_factors']['volatility_weight']
            vol_adjustment = 1 + (volatility * vol_weight)
            
            min_multiplier = self.config['trading']['adaptive_parameters']['stop_loss']['min_sl_multiplier']
            max_multiplier = self.config['trading']['adaptive_parameters']['stop_loss']['max_sl_multiplier']
            
            adjusted_multiplier = base_multiplier * vol_adjustment
            return np.clip(adjusted_multiplier, min_multiplier, max_multiplier)
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive SL multiplier: {str(e)}")
            raise

class TradeExecutor:
    def __init__(
        self,
        config: Dict[str, Any],
        client: AsyncBybitClient,
        risk_manager: RiskManager
    ):
        self.config = config
        self.client = client
        self.risk_manager = risk_manager
        self.logger = logging.getLogger(__name__)
        
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        market_data: pd.DataFrame,
        strategy_name: str
    ) -> Optional[TradePosition]:
        try:
            # Calculate position size
            position_size = await self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss=await self.risk_manager.calculate_stop_loss(
                    symbol=symbol,
                    entry_price=entry_price,
                    side=side,
                    market_data=market_data
                )
            )
            
            # Calculate stop loss and take profit
            stop_loss = await self.risk_manager.calculate_stop_loss(
                symbol=symbol,
                entry_price=entry_price,
                side=side,
                market_data=market_data
            )
            
            take_profit = self._calculate_take_profit(
                entry_price=entry_price,
                stop_loss=stop_loss,
                side=side
            )
            
            # Create and place order
            order = OrderData(
                symbol=symbol,
                side=side,
                order_type="Limit",
                qty=position_size,
                price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            order_result = await self.client.place_order(order)
            
            if order_result.get('order_id'):
                position = TradePosition(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_time=datetime.now(),
                    position_id=order_result['order_id'],
                    status='open'
                )
                
                self.logger.info(f"Trade executed: {asdict(position)}")
                return position
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            raise
            
    def _calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str
    ) -> float:
        try:
            risk = abs(entry_price - stop_loss)
            base_tp_ratio = self.config['trading']['adaptive_parameters']['profit_targets']['base_tp_ratio']
            
            if side == "Buy":
                take_profit = entry_price + (risk * base_tp_ratio)
            else:
                take_profit = entry_price - (risk * base_tp_ratio)
                
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            raise
            
    async def close_position(self, position: TradePosition) -> bool:
        try:
            result = await self.client.close_position(
                symbol=position.symbol,
                side=position.side
            )
            
            if result.get('order_id'):
                position.status = 'closed'
                self.logger.info(f"Position closed: {asdict(position)}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            raise
            
    async def modify_position(
        self,
        position: TradePosition,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None
    ) -> bool:
        try:
            if new_stop_loss or new_take_profit:
                order = OrderData(
                    symbol=position.symbol,
                    side=position.side,
                    order_type="Limit",
                    qty=position.size,
                    price=None,
                    stop_loss=new_stop_loss if new_stop_loss else position.stop_loss,
                    take_profit=new_take_profit if new_take_profit else position.take_profit,
                    reduce_only=True
                )
                
                result = await self.client.place_order(order)
                
                if result.get('order_id'):
                    if new_stop_loss:
                        position.stop_loss = new_stop_loss
                    if new_take_profit:
                        position.take_profit = new_take_profit
                        
                    self.logger.info(f"Position modified: {asdict(position)}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error modifying position: {str(e)}")
            raise
