import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"

class StrategyType(Enum):
    VWAP_PULLBACK = "vwap_pullback"
    VWAP_BREAKOUT = "vwap_breakout"

class TradingRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    UNKNOWN = "unknown"

class StrategyEngine:
    """
    VWAP-led strategy engine with pullback and breakout strategies
    Implements regime detection and multi-indicator confirmation
    """
    
    def __init__(self, indicator_engine, risk_manager):
        self.indicator_engine = indicator_engine
        self.risk_manager = risk_manager
        
        # Strategy configurations
        self.config = {
            'vwap_pullback': {
                'vwap_threshold': 0.001,  # 0.1% threshold for VWAP proximity
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'volume_factor': 1.2,  # Volume should be 20% above average
                'ema_confirmation': True
            },
            'vwap_breakout': {
                'vwap_threshold': 0.005,  # 0.5% threshold for breakout
                'volume_confirmation': 2.0,  # Volume should be 2x above average
                'rsi_range': (40, 60),  # RSI should be in middle range
                'macd_confirmation': True
            },
            'regime_detection': {
                'trend_threshold': 0.02,  # 2% price movement for trend detection
                'volatility_lookback': 20,
                'trend_lookback': 50
            }
        }
        
        self.active_signals = []
        self.performance_stats = {
            'total_signals': 0,
            'profitable_signals': 0,
            'loss_signals': 0,
            'win_rate': 0.0
        }
    
    def analyze_market_regime(self, ohlcv_data: List[Dict]) -> TradingRegime:
        """
        Detect market regime (trending vs ranging) to adjust strategy emphasis
        """
        try:
            if len(ohlcv_data) < self.config['regime_detection']['trend_lookback']:
                return TradingRegime.UNKNOWN
            
            df = pd.DataFrame(ohlcv_data)
            
            # Calculate price momentum
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-self.config['regime_detection']['trend_lookback']]) / df['close'].iloc[-self.config['regime_detection']['trend_lookback']]
            
            # Calculate volatility (ATR-based)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            atr = ranges.max(axis=1).rolling(self.config['regime_detection']['volatility_lookback']).mean()
            
            current_volatility = atr.iloc[-1] / df['close'].iloc[-1]  # Normalize by price
            avg_volatility = atr.mean() / df['close'].mean()
            
            # Regime classification
            if abs(price_change) > self.config['regime_detection']['trend_threshold']:
                return TradingRegime.TRENDING
            elif current_volatility < avg_volatility * 0.8:  # Low volatility
                return TradingRegime.RANGING
            else:
                return TradingRegime.TRENDING
                
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return TradingRegime.UNKNOWN
    
    def generate_signals(
        self,
        pair: str,
        ohlcv_data: List[Dict],
        strategy_type: StrategyType = StrategyType.VWAP_PULLBACK
    ) -> Dict[str, Any]:
        """
        Generate trading signals based on VWAP strategies with multi-indicator confirmation
        """
        try:
            if len(ohlcv_data) < 50:  # Need enough data for indicators
                return self._empty_signal_response()
            
            # Calculate all indicators
            indicators = self.indicator_engine.calculate_all_indicators(ohlcv_data)
            
            if not indicators:
                return self._empty_signal_response()
            
            # Detect market regime
            regime = self.analyze_market_regime(ohlcv_data)
            
            # Get current market data
            current_price = ohlcv_data[-1]['close']
            current_volume = ohlcv_data[-1]['volume']
            avg_volume = np.mean([candle['volume'] for candle in ohlcv_data[-20:]])  # 20-period average
            
            # Generate signals based on strategy type
            if strategy_type == StrategyType.VWAP_PULLBACK:
                signal_result = self._generate_vwap_pullback_signal(
                    current_price, current_volume, avg_volume, indicators, regime
                )
            elif strategy_type == StrategyType.VWAP_BREAKOUT:
                signal_result = self._generate_vwap_breakout_signal(
                    current_price, current_volume, avg_volume, indicators, regime
                )
            else:
                signal_result = self._empty_signal_response()
            
            # Add regime and indicator context
            signal_result['regime'] = regime.value
            signal_result['indicators'] = indicators
            signal_result['pair'] = pair
            signal_result['timestamp'] = datetime.utcnow().isoformat()
            
            # Update performance tracking
            self.performance_stats['total_signals'] += 1
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error generating signals for {pair}: {e}")
            return self._empty_signal_response()
    
    def _generate_vwap_pullback_signal(
        self,
        current_price: float,
        current_volume: float,
        avg_volume: float,
        indicators: Dict,
        regime: TradingRegime
    ) -> Dict[str, Any]:
        """
        VWAP Pullback Strategy:
        - Long: Price pulls back to VWAP in uptrend, RSI oversold, EMA alignment bullish
        - Short: Price pulls back to VWAP in downtrend, RSI overbought, EMA alignment bearish
        """
        
        vwap_data = indicators.get('vwap', {})
        rsi_data = indicators.get('rsi', {})
        ema_data = indicators.get('ema', {})
        
        vwap_value = vwap_data.get('vwap')
        rsi_value = rsi_data.get('rsi')
        ema_alignment = ema_data.get('ema_alignment')
        
        if not all([vwap_value, rsi_value, ema_alignment]):
            return self._empty_signal_response()
        
        # Calculate VWAP proximity
        vwap_distance = abs(current_price - vwap_value) / vwap_value
        
        signal = SignalType.HOLD
        confidence = 0.0
        reasons = []
        
        # Check for pullback opportunity
        if vwap_distance <= self.config['vwap_pullback']['vwap_threshold']:
            
            # Long signal conditions
            if (current_price > vwap_value and  # Above VWAP
                ema_alignment == 'bullish' and  # EMA alignment bullish
                rsi_value < self.config['vwap_pullback']['rsi_oversold'] and  # RSI oversold
                current_volume > avg_volume * self.config['vwap_pullback']['volume_factor']):  # Volume confirmation
                
                signal = SignalType.BUY
                confidence = self._calculate_confidence([
                    ('vwap_proximity', 0.3),
                    ('ema_alignment', 0.25),
                    ('rsi_oversold', 0.25),
                    ('volume_confirmation', 0.2)
                ])
                reasons = ['VWAP pullback', 'EMA bullish alignment', 'RSI oversold', 'Volume increase']
            
            # Short signal conditions  
            elif (current_price < vwap_value and  # Below VWAP
                  ema_alignment == 'bearish' and  # EMA alignment bearish
                  rsi_value > self.config['vwap_pullback']['rsi_overbought'] and  # RSI overbought
                  current_volume > avg_volume * self.config['vwap_pullback']['volume_factor']):  # Volume confirmation
                
                signal = SignalType.SELL
                confidence = self._calculate_confidence([
                    ('vwap_proximity', 0.3),
                    ('ema_alignment', 0.25),
                    ('rsi_overbought', 0.25),
                    ('volume_confirmation', 0.2)
                ])
                reasons = ['VWAP pullback', 'EMA bearish alignment', 'RSI overbought', 'Volume increase']
        
        # Adjust confidence based on regime
        if regime == TradingRegime.TRENDING:
            confidence *= 1.2  # Higher confidence in trending markets
        elif regime == TradingRegime.RANGING:
            confidence *= 0.8  # Lower confidence in ranging markets
        
        return {
            'signal': signal.value,
            'confidence': min(confidence, 1.0),  # Cap at 100%
            'strategy': 'vwap_pullback',
            'reasons': reasons,
            'entry_price': current_price,
            'vwap_distance': vwap_distance,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0
        }
    
    def _generate_vwap_breakout_signal(
        self,
        current_price: float,
        current_volume: float,
        avg_volume: float,
        indicators: Dict,
        regime: TradingRegime
    ) -> Dict[str, Any]:
        """
        VWAP Breakout Strategy:
        - Long: Price breaks above VWAP with volume, EMA alignment, MACD confirmation
        - Short: Price breaks below VWAP with volume, EMA alignment, MACD confirmation
        """
        
        vwap_data = indicators.get('vwap', {})
        macd_data = indicators.get('macd', {})
        ema_data = indicators.get('ema', {})
        rsi_data = indicators.get('rsi', {})
        
        vwap_value = vwap_data.get('vwap')
        macd_crossover = macd_data.get('macd_crossover')
        ema_alignment = ema_data.get('ema_alignment')
        rsi_value = rsi_data.get('rsi')
        
        if not all([vwap_value, macd_crossover, ema_alignment, rsi_value]):
            return self._empty_signal_response()
        
        # Calculate VWAP breakout distance
        vwap_distance = (current_price - vwap_value) / vwap_value
        
        signal = SignalType.HOLD
        confidence = 0.0
        reasons = []
        
        # Check for breakout conditions
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        rsi_in_range = self.config['vwap_breakout']['rsi_range'][0] <= rsi_value <= self.config['vwap_breakout']['rsi_range'][1]
        
        # Long breakout signal
        if (vwap_distance > self.config['vwap_breakout']['vwap_threshold'] and  # Price above VWAP threshold
            volume_ratio > self.config['vwap_breakout']['volume_confirmation'] and  # Volume confirmation
            ema_alignment == 'bullish' and  # EMA alignment
            macd_crossover == 'bullish' and  # MACD confirmation
            rsi_in_range):  # RSI not extreme
            
            signal = SignalType.BUY
            confidence = self._calculate_confidence([
                ('vwap_breakout', 0.35),
                ('volume_confirmation', 0.25),
                ('ema_alignment', 0.2),
                ('macd_confirmation', 0.2)
            ])
            reasons = ['VWAP upward breakout', 'High volume', 'EMA bullish', 'MACD bullish']
        
        # Short breakout signal
        elif (vwap_distance < -self.config['vwap_breakout']['vwap_threshold'] and  # Price below VWAP threshold
              volume_ratio > self.config['vwap_breakout']['volume_confirmation'] and  # Volume confirmation
              ema_alignment == 'bearish' and  # EMA alignment
              macd_crossover == 'bearish' and  # MACD confirmation
              rsi_in_range):  # RSI not extreme
            
            signal = SignalType.SELL
            confidence = self._calculate_confidence([
                ('vwap_breakout', 0.35),
                ('volume_confirmation', 0.25),
                ('ema_alignment', 0.2),
                ('macd_confirmation', 0.2)
            ])
            reasons = ['VWAP downward breakout', 'High volume', 'EMA bearish', 'MACD bearish']
        
        # Adjust confidence based on regime
        if regime == TradingRegime.TRENDING:
            confidence *= 1.3  # Higher confidence for breakouts in trending markets
        elif regime == TradingRegime.RANGING:
            confidence *= 0.6  # Much lower confidence in ranging markets
        
        return {
            'signal': signal.value,
            'confidence': min(confidence, 1.0),
            'strategy': 'vwap_breakout',
            'reasons': reasons,
            'entry_price': current_price,
            'vwap_distance': abs(vwap_distance),
            'volume_ratio': volume_ratio
        }
    
    def _calculate_confidence(self, factors: List[Tuple[str, float]]) -> float:
        """Calculate confidence score based on multiple factors"""
        return sum(weight for _, weight in factors)
    
    def _empty_signal_response(self) -> Dict[str, Any]:
        """Return empty signal response"""
        return {
            'signal': SignalType.HOLD.value,
            'confidence': 0.0,
            'strategy': 'none',
            'reasons': [],
            'entry_price': None,
            'vwap_distance': 0.0,
            'volume_ratio': 1.0
        }
    
    def validate_signal(self, signal_data: Dict, risk_params: Dict) -> Dict[str, Any]:
        """Validate signal against risk management parameters"""
        try:
            # Use risk manager to validate
            validation_result = self.risk_manager.validate_trade(signal_data, risk_params)
            
            return {
                'valid': validation_result['valid'],
                'reason': validation_result.get('reason', ''),
                'adjusted_size': validation_result.get('position_size', 0),
                'stop_loss': validation_result.get('stop_loss', None),
                'take_profit': validation_result.get('take_profit', None)
            }
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return {'valid': False, 'reason': f'Validation error: {e}'}
    
    def update_performance(self, signal_id: str, pnl: float):
        """Update performance statistics"""
        try:
            if pnl > 0:
                self.performance_stats['profitable_signals'] += 1
            else:
                self.performance_stats['loss_signals'] += 1
            
            total = self.performance_stats['profitable_signals'] + self.performance_stats['loss_signals']
            if total > 0:
                self.performance_stats['win_rate'] = self.performance_stats['profitable_signals'] / total
            
            logger.info(f"Performance updated - Win Rate: {self.performance_stats['win_rate']:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def get_strategy_config(self) -> Dict:
        """Get current strategy configuration"""
        return self.config.copy()
    
    def update_strategy_config(self, new_config: Dict):
        """Update strategy configuration"""
        self.config.update(new_config)
        logger.info("Strategy configuration updated")