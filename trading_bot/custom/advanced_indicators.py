# trading_bot/custom/advanced_indicators.py
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import logging
from .indicators import CustomIndicator, IndicatorResult

class OrderFlowAnalysis(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.volume_zones = params.get('volume_zones', 20)
        self.delta_threshold = params.get('delta_threshold', 0.6)
        self.imbalance_threshold = params.get('imbalance_threshold', 0.7)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Calculate order flow metrics
            delta = self._calculate_delta(data)
            imbalance = self._calculate_imbalance(data)
            footprint = self._create_footprint_map(data)
            liquidity_levels = self._find_liquidity_levels(data)
            absorption = self._detect_absorption(data)
            
            # Combine signals
            signal = self._generate_signal(delta, imbalance, absorption)
            strength = self._calculate_strength(delta, imbalance, absorption)
            
            return IndicatorResult(
                values=delta,
                signal=signal,
                strength=strength,
                additional_data={
                    'delta': delta,
                    'imbalance': imbalance,
                    'footprint': footprint,
                    'liquidity_levels': liquidity_levels,
                    'absorption': absorption
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in OrderFlowAnalysis: {str(e)}")
            raise
            
    def _calculate_delta(self, data: pd.DataFrame) -> pd.Series:
        """Calculate buying/selling pressure"""
        delta = pd.Series(index=data.index, dtype=float)
        
        # Determine aggressive orders
        delta = np.where(
            data['close'] > data['open'],
            data['volume'] * (1 + (data['close'] - data['open']) / data['open']),
            -data['volume'] * (1 + (data['open'] - data['close']) / data['close'])
        )
        
        return pd.Series(delta, index=data.index)
        
    def _calculate_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        buy_volume = np.where(data['close'] > data['open'], data['volume'], 0)
        sell_volume = np.where(data['close'] < data['open'], data['volume'], 0)
        
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        return pd.Series(imbalance, index=data.index)
        
    def _create_footprint_map(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price x volume footprint map"""
        footprint = pd.DataFrame()
        
        # Create price levels
        price_range = data['high'].max() - data['low'].min()
        price_step = price_range / self.volume_zones
        price_levels = np.arange(
            data['low'].min(),
            data['high'].max() + price_step,
            price_step
        )
        
        for i in range(len(data)):
            candle = data.iloc[i]
            price_range = np.arange(candle['low'], candle['high'], price_step)
            volume_per_level = candle['volume'] / len(price_range)
            
            for price in price_range:
                level_idx = np.digitize(price, price_levels) - 1
                if 0 <= level_idx < len(price_levels):
                    if i not in footprint.columns:
                        footprint[i] = 0
                    footprint.at[level_idx, i] = volume_per_level
                    
        return footprint
        
    def _find_liquidity_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key liquidity levels"""
        footprint = self._create_footprint_map(data)
        
        # Find high volume nodes
        volume_threshold = footprint.sum().mean() + footprint.sum().std()
        high_volume_levels = footprint[footprint.sum(axis=1) > volume_threshold].index
        
        # Separate buy and sell liquidity
        buy_liquidity = []
        sell_liquidity = []
        
        for level in high_volume_levels:
            if footprint.iloc[level].mean() > 0:
                if level < len(footprint) // 2:
                    buy_liquidity.append(level)
                else:
                    sell_liquidity.append(level)
                    
        return {
            'buy_liquidity': buy_liquidity,
            'sell_liquidity': sell_liquidity
        }
        
    def _detect_absorption(self, data: pd.DataFrame) -> pd.Series:
        """Detect volume absorption"""
        absorption = pd.Series(index=data.index, dtype=float)
        
        for i in range(1, len(data)):
            curr_candle = data.iloc[i]
            prev_candle = data.iloc[i-1]
            
            # Check for absorption conditions
            high_volume = curr_candle['volume'] > data['volume'].rolling(20).mean().iloc[i]
            small_range = (curr_candle['high'] - curr_candle['low']) < \
                         (data['high'] - data['low']).rolling(20).std().iloc[i]
            
            if high_volume and small_range:
                if curr_candle['close'] > prev_candle['close']:
                    absorption.iloc[i] = 1  # Bullish absorption
                else:
                    absorption.iloc[i] = -1  # Bearish absorption
                    
        return absorption
        
    def _generate_signal(
        self,
        delta: pd.Series,
        imbalance: pd.Series,
        absorption: pd.Series
    ) -> str:
        """Generate trading signal based on order flow analysis"""
        recent_delta = delta.tail(5).mean()
        recent_imbalance = imbalance.tail(5).mean()
        recent_absorption = absorption.tail(5).mean()
        
        if (recent_delta > self.delta_threshold and
            recent_imbalance > self.imbalance_threshold and
            recent_absorption > 0):
            return 'strong_buy'
        elif (recent_delta < -self.delta_threshold and
              recent_imbalance < -self.imbalance_threshold and
              recent_absorption < 0):
            return 'strong_sell'
        elif recent_delta > 0 and recent_imbalance > 0:
            return 'buy'
        elif recent_delta < 0 and recent_imbalance < 0:
            return 'sell'
        return 'neutral'
        
    def _calculate_strength(
        self,
        delta: pd.Series,
        imbalance: pd.Series,
        absorption: pd.Series
    ) -> float:
        """Calculate signal strength based on order flow metrics"""
        recent_delta = abs(delta.tail(5).mean())
        recent_imbalance = abs(imbalance.tail(5).mean())
        recent_absorption = abs(absorption.tail(5).mean())
        
        strength = (
            0.4 * recent_delta +
            0.4 * recent_imbalance +
            0.2 * recent_absorption
        )
        
        return min(max(strength, 0), 1)

class MarketRegimeAnalysis(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.lookback = params.get('lookback', 100)
        self.volatility_window = params.get('volatility_window', 20)
        self.regime_threshold = params.get('regime_threshold', 0.6)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Calculate regime components
            volatility = self._calculate_volatility(data)
            trend = self._calculate_trend(data)
            momentum = self._calculate_momentum(data)
            
            # Identify market regime
            regime = self._identify_regime(volatility[-20:], trend[-20:], momentum[-20:])
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(regime, volatility[-1])
            
            # Generate trading signal
            signal = self._generate_signal(regime)
            
            return IndicatorResult(
                values=np.array([signal_strength]),
                signal=signal,
                strength=signal_strength,
                additional_data={
                    'regime': regime,
                    'volatility': volatility,
                    'trend': trend,
                    'momentum': momentum
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in MarketRegimeAnalysis calculation: {str(e)}")
            raise
            
    def _calculate_volatility(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate normalized volatility"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(self.volatility_window).std()
        normalized_vol = (volatility - volatility.min()) / (volatility.max() - volatility.min())
        return normalized_vol.fillna(0).values
        
    def _calculate_trend(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate trend strength"""
        ema_short = ta.ema(data['close'], length=20)
        ema_long = ta.ema(data['close'], length=50)
        trend = (ema_short - ema_long) / ema_long
        return trend.fillna(0).values
        
    def _calculate_momentum(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate momentum"""
        rsi = ta.rsi(data['close'], length=14)
        roc = ta.roc(data['close'], length=10)
        momentum = (rsi/100 + roc) / 2
        return momentum.fillna(0).values
        
    def _identify_regime(self, volatility: np.ndarray, trend: np.ndarray, momentum: np.ndarray) -> str:
        """Identify current market regime"""
        vol_level = np.mean(volatility)
        trend_strength = abs(np.mean(trend))
        mom_strength = abs(np.mean(momentum))
        
        if vol_level > self.regime_threshold:
            if trend_strength > self.regime_threshold:
                return "volatile_trending"
            return "volatile_ranging"
        else:
            if trend_strength > self.regime_threshold:
                return "stable_trending"
            return "stable_ranging"
            
    def _calculate_signal_strength(self, regime: str, current_volatility: float) -> float:
        """Calculate signal strength based on regime and volatility"""
        base_strength = {
            "volatile_trending": 0.8,
            "volatile_ranging": 0.4,
            "stable_trending": 1.0,
            "stable_ranging": 0.6
        }.get(regime, 0.5)
        
        # Adjust strength based on volatility
        vol_adjustment = 1 - (current_volatility * 0.5)  # Reduce strength in high volatility
        
        return min(max(base_strength * vol_adjustment, 0), 1)
        
    def _generate_signal(self, regime: str) -> str:
        """Generate trading signal based on regime"""
        signal_map = {
            "volatile_trending": "neutral",
            "volatile_ranging": "hold",
            "stable_trending": "buy",
            "stable_ranging": "neutral"
        }
        return signal_map.get(regime, "hold")

# Initialize indicators
def initialize_advanced_indicators(config: Dict[str, Any]) -> Dict[str, CustomIndicator]:
    """Initialize all advanced indicators"""
    indicators = {
        'order_flow': OrderFlowAnalysis(config),
        'market_regime': MarketRegimeAnalysis(config)
    }
    return indicators