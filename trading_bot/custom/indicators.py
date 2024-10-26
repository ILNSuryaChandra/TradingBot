# trading_bot/custom/indicators.py
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

@dataclass
class IndicatorResult:
    values: np.ndarray
    signal: Optional[str] = None
    strength: float = 0.0
    additional_data: Dict[str, Any] = None

class CustomIndicator(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        pass
        
    def validate_params(self) -> bool:
        return True

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

class VolatilityIndicator(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.window = params.get('window', 20)
        self.atr_multiplier = params.get('atr_multiplier', 1.5)
        self.std_window = params.get('std_window', 20)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Calculate ATR
            atr = ta.atr(data['high'], data['low'], data['close'], length=self.window)
            
            # Calculate standard deviation of returns
            returns = data['close'].pct_change()
            std = returns.rolling(window=self.std_window).std()
            
            # Combine indicators
            volatility = (atr / data['close'] + std) / 2
            
            # Generate signal based on volatility levels
            signal = self._generate_signal(volatility.iloc[-1])
            strength = self._calculate_strength(volatility.iloc[-1])
            
            return IndicatorResult(
                values=volatility.values,
                signal=signal,
                strength=strength,
                additional_data={
                    'atr': atr,
                    'std': std
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in VolatilityIndicator calculation: {str(e)}")
            raise
            
    def _generate_signal(self, current_volatility: float) -> str:
        if current_volatility > self.atr_multiplier * 2:
            return "high_volatility"
        elif current_volatility > self.atr_multiplier:
            return "medium_volatility"
        return "low_volatility"
        
    def _calculate_strength(self, current_volatility: float) -> float:
        return min(current_volatility / (self.atr_multiplier * 2), 1.0)

class MomentumIndicator(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.rsi_period = params.get('rsi_period', 14)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Calculate RSI
            rsi = ta.rsi(data['close'], length=self.rsi_period)
            
            # Calculate MACD
            macd = ta.macd(
                data['close'],
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal
            )
            
            # Calculate momentum
            momentum = data['close'].pct_change(self.rsi_period)
            
            # Combine indicators
            composite_momentum = (
                0.4 * (rsi / 100) +
                0.4 * (macd['MACD_12_26_9'] / data['close']) +
                0.2 * momentum
            )
            
            # Generate signal
            signal = self._generate_signal(composite_momentum.iloc[-1])
            strength = abs(composite_momentum.iloc[-1])
            
            return IndicatorResult(
                values=composite_momentum.values,
                signal=signal,
                strength=min(strength, 1.0),
                additional_data={
                    'rsi': rsi,
                    'macd': macd,
                    'momentum': momentum
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum index: {str(e)}")
            raise
            
    def _generate_signal(self, value: float) -> str:
        if value > 0.7:
            return 'strong_buy'
        elif value > 0.3:
            return 'buy'
        elif value < -0.7:
            return 'strong_sell'
        elif value < -0.3:
            return 'sell'
        return 'neutral'

class VolumeProfileIndicator(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.volume_zones = params.get('volume_zones', 20)
        self.delta_threshold = params.get('delta_threshold', 0.6)
        self.imbalance_threshold = params.get('imbalance_threshold', 0.7)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Calculate volume profile
            profile = self._calculate_volume_profile(data)
            
            # Calculate volume delta
            delta = self._calculate_delta(data)
            
            # Generate signal
            signal = self._generate_signal(delta)
            strength = self._calculate_strength(delta)
            
            return IndicatorResult(
                values=delta.values,
                signal=signal,
                strength=strength,
                additional_data={
                    'profile': profile,
                    'delta': delta
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in VolumeProfileIndicator calculation: {str(e)}")
            raise
            
    def _calculate_volume_profile(self, data: pd.DataFrame) -> pd.DataFrame:
        price_range = data['high'].max() - data['low'].min()
        zone_size = price_range / self.volume_zones
        
        profile = pd.DataFrame()
        profile['price_level'] = np.arange(
            data['low'].min(),
            data['high'].max(),
            zone_size
        )
        profile['volume'] = 0
        
        for idx, row in data.iterrows():
            zone = int((row['close'] - data['low'].min()) / zone_size)
            if 0 <= zone < len(profile):
                profile.loc[zone, 'volume'] += row['volume']
                
        return profile
        
    def _calculate_delta(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(
            np.where(
                data['close'] > data['open'],
                data['volume'],
                -data['volume']
            ),
            index=data.index
        )
        
    def _generate_signal(self, delta: pd.Series) -> str:
        recent_delta = delta.tail(20).mean()
        if abs(recent_delta) > self.delta_threshold * delta.std():
            return "buy" if recent_delta > 0 else "sell"
        return "neutral"
        
    def _calculate_strength(self, delta: pd.Series) -> float:
        recent_delta = delta.tail(20).mean()
        delta_strength = abs(recent_delta) / (delta.std() * self.delta_threshold)
        return min(delta_strength, 1.0)