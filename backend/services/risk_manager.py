import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from enum import Enum
import os

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskManager:
    """
    Risk management with position sizing, stop-loss, daily loss limits,
    and live-trading guardrails
    """
    
    def __init__(self):
        # Load risk parameters from environment
        self.config = {
            'max_daily_loss_percent': float(os.getenv('MAX_DAILY_LOSS_PERCENT', '2.0')),
            'max_position_size_percent': float(os.getenv('MAX_POSITION_SIZE_PERCENT', '10.0')),
            'default_stop_loss_percent': float(os.getenv('DEFAULT_STOP_LOSS_PERCENT', '1.0')),
            'max_concurrent_positions': 5,
            'max_leverage': 1.0,  # Start with no leverage
            'volatility_lookback': 20,
            'risk_free_rate': 0.02,  # 2% annual risk-free rate
            'max_drawdown_percent': 15.0,
            'position_correlation_limit': 0.7,
            'circuit_breaker_loss_percent': 5.0  # Emergency stop
        }
        
        # Track current state
        self.daily_pnl = 0.0
        self.current_positions = {}
        self.daily_trades = []
        self.total_capital = 10000.0  # Default capital
        self.available_capital = self.total_capital
        self.max_drawdown = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.emergency_stop_reason = None
        
        logger.info("Risk Manager initialized")
    
    def set_capital(self, capital: float):
        """Set total trading capital"""
        self.total_capital = capital
        self.available_capital = capital
        logger.info(f"Trading capital set to ${capital:,.2f}")
    
    def validate_trade(self, signal_data: Dict, market_data: Dict = None) -> Dict[str, Any]:
        """
        Validate trade against all risk parameters
        Returns validation result with position sizing
        """
        try:
            # Check circuit breaker
            if self.circuit_breaker_active:
                return {
                    'valid': False,
                    'reason': f'Circuit breaker active: {self.emergency_stop_reason}',
                    'risk_level': RiskLevel.CRITICAL.value
                }
            
            # Reset daily tracking if new day
            self._reset_daily_tracking()
            
            # Check daily loss limit
            if not self._check_daily_loss_limit():
                return {
                    'valid': False,
                    'reason': f'Daily loss limit exceeded: {self.daily_pnl:.2%}',
                    'risk_level': RiskLevel.CRITICAL.value
                }
            
            # Calculate position size
            position_size = self._calculate_position_size(signal_data, market_data)
            
            if position_size <= 0:
                return {
                    'valid': False,
                    'reason': 'Calculated position size is zero or negative',
                    'risk_level': RiskLevel.HIGH.value
                }
            
            # Check position limits
            if len(self.current_positions) >= self.config['max_concurrent_positions']:
                return {
                    'valid': False,
                    'reason': f'Maximum concurrent positions reached: {len(self.current_positions)}',
                    'risk_level': RiskLevel.MEDIUM.value
                }
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self._calculate_stop_take_levels(signal_data, market_data)
            
            # Calculate risk metrics
            risk_assessment = self._assess_trade_risk(signal_data, position_size, stop_loss)
            
            return {
                'valid': True,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_level': risk_assessment['risk_level'],
                'risk_metrics': risk_assessment,
                'max_loss': risk_assessment['max_loss_dollar'],
                'risk_reward_ratio': risk_assessment.get('risk_reward_ratio', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return {
                'valid': False,
                'reason': f'Risk validation error: {e}',
                'risk_level': RiskLevel.HIGH.value
            }
    
    def _calculate_position_size(self, signal_data: Dict, market_data: Dict = None) -> float:
        """
        Calculate position size using multiple methods:
        1. Fixed percentage method
        2. Volatility-adjusted method
        3. Kelly criterion (if sufficient data)
        """
        try:
            entry_price = signal_data.get('entry_price', 0)
            confidence = signal_data.get('confidence', 0)
            
            if entry_price <= 0:
                return 0
            
            # Method 1: Fixed percentage of capital
            fixed_size = (self.available_capital * self.config['max_position_size_percent'] / 100) / entry_price
            
            # Method 2: Volatility-adjusted sizing
            volatility_size = fixed_size
            if market_data and 'atr' in market_data:
                atr = market_data['atr']
                volatility_adjustment = min(2.0, max(0.5, 1.0 / (atr / entry_price)))
                volatility_size = fixed_size * volatility_adjustment
            
            # Method 3: Confidence-adjusted sizing
            confidence_adjusted_size = volatility_size * max(0.1, confidence)
            
            # Apply maximum position size limit
            max_position_value = self.available_capital * (self.config['max_position_size_percent'] / 100)
            max_position_size = max_position_value / entry_price
            
            final_size = min(confidence_adjusted_size, max_position_size)
            
            logger.debug(f"Position sizing - Fixed: {fixed_size:.4f}, Volatility: {volatility_size:.4f}, Final: {final_size:.4f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _calculate_stop_take_levels(self, signal_data: Dict, market_data: Dict = None) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            entry_price = signal_data.get('entry_price', 0)
            signal_type = signal_data.get('signal', 'hold')
            
            if entry_price <= 0:
                return None, None
            
            # Default stop loss percentage
            stop_loss_pct = self.config['default_stop_loss_percent'] / 100
            
            # Adjust based on ATR if available
            if market_data and 'atr' in market_data:
                atr = market_data['atr']
                atr_stop_pct = (atr * 2) / entry_price  # 2x ATR stop
                stop_loss_pct = max(stop_loss_pct, atr_stop_pct)
            
            # Calculate levels based on signal direction
            if signal_type == 'buy':
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + stop_loss_pct * 2)  # 2:1 reward-to-risk
            elif signal_type == 'sell':
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - stop_loss_pct * 2)
            else:
                return None, None
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stop/take levels: {e}")
            return None, None
    
    def _assess_trade_risk(self, signal_data: Dict, position_size: float, stop_loss: Optional[float]) -> Dict[str, Any]:
        """Assess overall risk level of the trade"""
        try:
            entry_price = signal_data.get('entry_price', 0)
            confidence = signal_data.get('confidence', 0)
            
            # Calculate maximum loss
            if stop_loss and entry_price > 0:
                max_loss_pct = abs(stop_loss - entry_price) / entry_price
                max_loss_dollar = position_size * entry_price * max_loss_pct
            else:
                max_loss_pct = self.config['default_stop_loss_percent'] / 100
                max_loss_dollar = position_size * entry_price * max_loss_pct
            
            # Risk as percentage of total capital
            capital_risk_pct = max_loss_dollar / self.total_capital
            
            # Determine risk level
            if capital_risk_pct > 0.05:  # > 5% of capital
                risk_level = RiskLevel.CRITICAL
            elif capital_risk_pct > 0.02:  # > 2% of capital
                risk_level = RiskLevel.HIGH
            elif capital_risk_pct > 0.01:  # > 1% of capital
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Adjust based on confidence
            if confidence < 0.3:
                risk_level = RiskLevel.HIGH
            
            return {
                'risk_level': risk_level.value,
                'max_loss_dollar': max_loss_dollar,
                'max_loss_percent': max_loss_pct,
                'capital_risk_percent': capital_risk_pct,
                'confidence_factor': confidence,
                'position_value': position_size * entry_price
            }
            
        except Exception as e:
            logger.error(f"Error assessing trade risk: {e}")
            return {'risk_level': RiskLevel.HIGH.value}
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded"""
        daily_loss_pct = abs(self.daily_pnl) / self.total_capital
        return daily_loss_pct < (self.config['max_daily_loss_percent'] / 100)
    
    def _reset_daily_tracking(self):
        """Reset daily tracking if it's a new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = []
            self.last_reset_date = current_date
            logger.info("Daily risk tracking reset")
    
    def update_position(self, position_id: str, position_data: Dict):
        """Update current position"""
        self.current_positions[position_id] = position_data
    
    def close_position(self, position_id: str, pnl: float):
        """Close position and update risk metrics"""
        try:
            if position_id in self.current_positions:
                position = self.current_positions.pop(position_id)
                
                # Update daily PnL
                self.daily_pnl += pnl
                
                # Update available capital
                self.available_capital += pnl
                
                # Check for circuit breaker conditions
                self._check_circuit_breaker()
                
                # Track trade
                self.daily_trades.append({
                    'position_id': position_id,
                    'pnl': pnl,
                    'timestamp': datetime.utcnow().isoformat(),
                    'position_data': position
                })
                
                logger.info(f"Position {position_id} closed. PnL: ${pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    def _check_circuit_breaker(self):
        """Check if circuit breaker should be activated"""
        try:
            # Check daily loss threshold
            daily_loss_pct = abs(self.daily_pnl) / self.total_capital
            
            if daily_loss_pct > (self.config['circuit_breaker_loss_percent'] / 100):
                self.circuit_breaker_active = True
                self.emergency_stop_reason = f"Daily loss exceeded {self.config['circuit_breaker_loss_percent']}%"
                logger.critical(f"CIRCUIT BREAKER ACTIVATED: {self.emergency_stop_reason}")
            
            # Check drawdown
            current_capital = self.total_capital + self.daily_pnl
            drawdown_pct = (self.total_capital - current_capital) / self.total_capital
            
            if drawdown_pct > (self.config['max_drawdown_percent'] / 100):
                self.circuit_breaker_active = True
                self.emergency_stop_reason = f"Max drawdown exceeded {self.config['max_drawdown_percent']}%"
                logger.critical(f"CIRCUIT BREAKER ACTIVATED: {self.emergency_stop_reason}")
                
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
    
    def reset_circuit_breaker(self, reason: str = "Manual reset"):
        """Reset circuit breaker (admin function)"""
        self.circuit_breaker_active = False
        self.emergency_stop_reason = None
        logger.warning(f"Circuit breaker reset: {reason}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk management summary"""
        self._reset_daily_tracking()
        
        current_capital = self.total_capital + self.daily_pnl
        utilization_pct = (self.total_capital - self.available_capital) / self.total_capital
        
        return {
            'total_capital': self.total_capital,
            'available_capital': self.available_capital,
            'current_capital': current_capital,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': self.daily_pnl / self.total_capital,
            'capital_utilization_percent': utilization_pct,
            'active_positions': len(self.current_positions),
            'max_positions': self.config['max_concurrent_positions'],
            'daily_trades_count': len(self.daily_trades),
            'circuit_breaker_active': self.circuit_breaker_active,
            'emergency_stop_reason': self.emergency_stop_reason,
            'risk_limits': {
                'max_daily_loss_percent': self.config['max_daily_loss_percent'],
                'max_position_size_percent': self.config['max_position_size_percent'],
                'max_drawdown_percent': self.config['max_drawdown_percent']
            }
        }
    
    def update_config(self, new_config: Dict):
        """Update risk management configuration"""
        self.config.update(new_config)
        logger.info("Risk management configuration updated")