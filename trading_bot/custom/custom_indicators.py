from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from .indicators import CustomIndicator, IndicatorResult

class CustomVolumeProfile(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.volume_zones = params.get('volume_zones', 20)
        self.value_area_volume = params.get('value_area_volume', 0.68)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Calculate price levels
            price_range = data['high'].max() - data['low'].min()
            zone_size = price_range / self.volume_zones
            
            # Create volume profile
            profile = pd.DataFrame()
            price_levels = np.arange(
                data['low'].min(),
                data['high'].max() + zone_size,
                zone_size
            )
            profile['price'] = price_levels[:-1]
            profile['volume'] = 0
            
            # Distribute volume across price levels
            for i in range(len(data)):
                candle = data.iloc[i]
                price_range = np.arange(candle['low'], candle['high'], zone_size)
                vol_per_level = candle['volume'] / len(price_range)
                
                for price in price_range:
                    level_idx = np.searchsorted(price_levels, price) - 1
                    if 0 <= level_idx < len(profile):
                        profile.loc[level_idx, 'volume'] += vol_per_level
            
            # Calculate POC (Point of Control)
            poc_idx = profile['volume'].idxmax()
            poc_price = profile.loc[poc_idx, 'price']
            
            # Calculate Value Area
            total_volume = profile['volume'].sum()
            threshold = total_volume * self.value_area_volume
            sorted_profile = profile.sort_values('volume', ascending=False)
            cumsum = sorted_profile['volume'].cumsum()
            value_area = sorted_profile[cumsum <= threshold]
            
            va_high = value_area['price'].max()
            va_low = value_area['price'].min()
            
            # Generate signal based on current price position
            current_price = data['close'].iloc[-1]
            if current_price > va_high:
                signal = 'overbought'
            elif current_price < va_low:
                signal = 'oversold'
            else:
                signal = 'neutral'
                
            # Calculate signal strength based on distance from POC
            strength = min(abs(current_price - poc_price) / (va_high - va_low), 1.0)
            
            return IndicatorResult(
                values=profile['volume'].values,
                signal=signal,
                strength=strength,
                additional_data={
                    'poc_price': poc_price,
                    'value_area_high': va_high,
                    'value_area_low': va_low,
                    'profile': profile.to_dict('records')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {str(e)}")
            raise

class CustomMarketStructure(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.window = params.get('window', 5)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Identify swing points
            highs = self._find_swing_highs(data)
            lows = self._find_swing_lows(data)
            
            # Analyze market structure
            trend = self._analyze_trend(data, highs, lows)
            structure = self._analyze_structure(data, highs, lows)
            strength = self._calculate_strength(data, trend, structure)
            
            return IndicatorResult(
                values=np.array([strength]),
                signal=trend,
                strength=strength,
                additional_data={
                    'structure': structure,
                    'swing_highs': highs,
                    'swing_lows': lows
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {str(e)}")
            raise
            
    def _find_swing_highs(self, data: pd.DataFrame) -> List[int]:
        """Find swing high points"""
        highs = []
        for i in range(self.window, len(data) - self.window):
            if all(data['high'].iloc[i] > data['high'].iloc[i-j] for j in range(1, self.window+1)) and \
               all(data['high'].iloc[i] > data['high'].iloc[i+j] for j in range(1, self.window+1)):
                highs.append(i)
        return highs
        
    def _find_swing_lows(self, data: pd.DataFrame) -> List[int]:
        """Find swing low points"""
        lows = []
        for i in range(self.window, len(data) - self.window):
            if all(data['low'].iloc[i] < data['low'].iloc[i-j] for j in range(1, self.window+1)) and \
               all(data['low'].iloc[i] < data['low'].iloc[i+j] for j in range(1, self.window+1)):
                lows.append(i)
        return lows
        
    def _analyze_trend(self, data: pd.DataFrame, highs: List[int], lows: List[int]) -> str:
        if not highs or not lows:
            return 'neutral'
            
        # Get recent swing points
        recent_highs = data['high'].iloc[highs[-2:]]
        recent_lows = data['low'].iloc[lows[-2:]]
        
        # Determine trend based on swing points
        if len(recent_highs) > 1 and len(recent_lows) > 1:
            higher_highs = recent_highs.iloc[1] > recent_highs.iloc[0]
            higher_lows = recent_lows.iloc[1] > recent_lows.iloc[0]
            
            if higher_highs and higher_lows:
                return 'uptrend'
            elif not higher_highs and not higher_lows:
                return 'downtrend'
                
        return 'neutral'
        
    def _analyze_structure(self, data: pd.DataFrame, highs: List[int], lows: List[int]) -> Dict[str, Any]:
        return {
            'swing_high_prices': data['high'].iloc[highs].tolist() if highs else [],
            'swing_low_prices': data['low'].iloc[lows].tolist() if lows else [],
            'num_swings': len(highs) + len(lows)
        }
        
    def _calculate_strength(self, data: pd.DataFrame, trend: str, structure: Dict[str, Any]) -> float:
        if trend == 'neutral':
            return 0.5
            
        # Calculate strength based on swing point consistency
        if not structure['swing_high_prices'] or not structure['swing_low_prices']:
            return 0.0
            
        # Calculate trend consistency
        recent_swings = min(len(structure['swing_high_prices']), len(structure['swing_low_prices']))
        consistent_swings = 0
        
        for i in range(1, recent_swings):
            if trend == 'uptrend':
                if structure['swing_high_prices'][i] > structure['swing_high_prices'][i-1] and \
                   structure['swing_low_prices'][i] > structure['swing_low_prices'][i-1]:
                    consistent_swings += 1
            else:  # downtrend
                if structure['swing_high_prices'][i] < structure['swing_high_prices'][i-1] and \
                   structure['swing_low_prices'][i] < structure['swing_low_prices'][i-1]:
                    consistent_swings += 1
                    
        return consistent_swings / (recent_swings - 1) if recent_swings > 1 else 0.0

class CustomMomentumIndex(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.rsi_period = params.get('rsi_period', 14)
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Calculate components
            rsi = ta.rsi(data['close'], length=self.rsi_period)
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
            signal = self._generate_signal(
                rsi.iloc[-1],
                macd['MACD_12_26_9'].iloc[-1],
                macd['MACDs_12_26_9'].iloc[-1],
                momentum.iloc[-1]
            )
            
            # Calculate strength
            strength = self._calculate_strength(
                rsi.iloc[-1],
                macd['MACD_12_26_9'].iloc[-1],
                macd['MACDs_12_26_9'].iloc[-1],
                momentum.iloc[-1]
            )
            
            return IndicatorResult(
                values=composite_momentum.values,
                signal=signal,
                strength=strength,
                additional_data={
                    'rsi': rsi.iloc[-1],
                    'macd': macd['MACD_12_26_9'].iloc[-1],
                    'macd_signal': macd['MACDs_12_26_9'].iloc[-1],
                    'momentum': momentum.iloc[-1]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum index: {str(e)}")
            raise
            
    def _generate_signal(
        self,
        rsi: float,
        macd: float,
        macd_signal: float,
        momentum: float
    ) -> str:
        # RSI conditions
        rsi_overbought = rsi > 70
        rsi_oversold = rsi < 30
        
        # MACD conditions
        macd_bullish = macd > macd_signal
        macd_bearish = macd < macd_signal
        
        # Momentum conditions
        strong_momentum = abs(momentum) > 0.02
        
        # Generate signal
        if rsi_oversold and macd_bullish and momentum > 0:
            return 'strong_buy'
        elif rsi_overbought and macd_bearish and momentum < 0:
            return 'strong_sell'
        elif macd_bullish and momentum > 0:
            return 'buy'
        elif macd_bearish and momentum < 0:
            return 'sell'
        return 'neutral'
        
    def _calculate_strength(
        self,
        rsi: float,
        macd: float,
        macd_signal: float,
        momentum: float
    ) -> float:
        # RSI strength
        rsi_strength = abs(rsi - 50) / 50
        
        # MACD strength
        macd_strength = abs(macd - macd_signal) / abs(macd_signal) if macd_signal != 0 else 0
        
        # Momentum strength
        momentum_strength = min(abs(momentum) / 0.02, 1.0)
        
        # Combine strengths
        total_strength = (
            0.4 * rsi_strength +
            0.4 * macd_strength +
            0.2 * momentum_strength
        )
        
        return min(max(total_strength, 0), 1)