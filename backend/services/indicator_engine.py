import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class IndicatorEngine:
    """
    Fast, vectorized indicators: VWAP, RSI, EMA, MACD, CCI, OBV, Bollinger, PSAR
    Returns both raw values and normalized signals
    """
    
    def __init__(self):
        self.config = {
            'vwap': {'session_reset': True, 'std_bands': True},
            'rsi': {'period': 14, 'oversold': 30, 'overbought': 70},
            'ema': {'fast': 9, 'medium': 21, 'slow': 50},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'cci': {'period': 20, 'oversold': -100, 'overbought': 100},
            'bollinger': {'period': 20, 'std': 2},
            'psar': {'af': 0.02, 'max_af': 0.2}
        }
    
    def calculate_all_indicators(self, ohlcv_data: List[Dict]) -> Dict[str, Any]:
        """Calculate all technical indicators for given OHLCV data"""
        try:
            if not ohlcv_data:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            indicators = {}
            
            # Calculate each indicator
            indicators['vwap'] = self.calculate_vwap(df)
            indicators['rsi'] = self.calculate_rsi(df)
            indicators['ema'] = self.calculate_ema(df)
            indicators['macd'] = self.calculate_macd(df)
            indicators['cci'] = self.calculate_cci(df)
            indicators['obv'] = self.calculate_obv(df)
            indicators['bollinger'] = self.calculate_bollinger_bands(df)
            indicators['psar'] = self.calculate_psar(df)
            
            # Calculate signals
            indicators['signals'] = self.calculate_signals(df, indicators)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def calculate_vwap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate VWAP with session reset and standard deviation bands"""
        try:
            # Typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Volume weighted price
            vwp = typical_price * df['volume']
            
            # VWAP calculation
            if self.config['vwap']['session_reset']:
                # Daily session VWAP reset
                df_copy = df.copy()
                df_copy['date'] = df_copy.index.date
                
                vwap_values = []
                for date in df_copy['date'].unique():
                    day_data = df_copy[df_copy['date'] == date]
                    day_typical = (day_data['high'] + day_data['low'] + day_data['close']) / 3
                    day_vwp = day_typical * day_data['volume']
                    day_vwap = day_vwp.cumsum() / day_data['volume'].cumsum()
                    vwap_values.extend(day_vwap.tolist())
                
                vwap = pd.Series(vwap_values, index=df.index)
            else:
                # Anchored VWAP
                vwap = vwp.cumsum() / df['volume'].cumsum()
            
            result = {
                'vwap': vwap.iloc[-1] if not vwap.empty else None,
                'vwap_series': vwap.tolist()
            }
            
            # Add standard deviation bands if enabled
            if self.config['vwap']['std_bands']:
                vwap_std = ((typical_price - vwap) ** 2 * df['volume']).cumsum() / df['volume'].cumsum()
                vwap_std = np.sqrt(vwap_std)
                
                result['vwap_upper'] = (vwap + vwap_std).iloc[-1] if not vwap.empty else None
                result['vwap_lower'] = (vwap - vwap_std).iloc[-1] if not vwap.empty else None
                result['vwap_std'] = vwap_std.iloc[-1] if not vwap_std.empty else None
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return {}
    
    def calculate_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI indicator"""
        try:
            rsi = ta.momentum.RSIIndicator(
                close=df['close'],
                window=self.config['rsi']['period']
            ).rsi()
            
            return {
                'rsi': rsi.iloc[-1] if not rsi.empty else None,
                'rsi_series': rsi.tolist(),
                'rsi_signal': self._get_rsi_signal(rsi.iloc[-1] if not rsi.empty else 50)
            }
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return {}
    
    def calculate_ema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate EMA indicators (9, 21, 50)"""
        try:
            ema_fast = ta.trend.EMAIndicator(close=df['close'], window=self.config['ema']['fast']).ema_indicator()
            ema_medium = ta.trend.EMAIndicator(close=df['close'], window=self.config['ema']['medium']).ema_indicator()
            ema_slow = ta.trend.EMAIndicator(close=df['close'], window=self.config['ema']['slow']).ema_indicator()
            
            return {
                'ema_9': ema_fast.iloc[-1] if not ema_fast.empty else None,
                'ema_21': ema_medium.iloc[-1] if not ema_medium.empty else None,
                'ema_50': ema_slow.iloc[-1] if not ema_slow.empty else None,
                'ema_9_series': ema_fast.tolist(),
                'ema_21_series': ema_medium.tolist(),
                'ema_50_series': ema_slow.tolist(),
                'ema_alignment': self._get_ema_alignment(
                    ema_fast.iloc[-1] if not ema_fast.empty else None,
                    ema_medium.iloc[-1] if not ema_medium.empty else None,
                    ema_slow.iloc[-1] if not ema_slow.empty else None
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return {}
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD indicator"""
        try:
            macd_indicator = ta.trend.MACD(
                close=df['close'],
                window_fast=self.config['macd']['fast'],
                window_slow=self.config['macd']['slow'],
                window_sign=self.config['macd']['signal']
            )
            
            macd_line = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_histogram = macd_indicator.macd_diff()
            
            return {
                'macd': macd_line.iloc[-1] if not macd_line.empty else None,
                'macd_signal': macd_signal.iloc[-1] if not macd_signal.empty else None,
                'macd_histogram': macd_histogram.iloc[-1] if not macd_histogram.empty else None,
                'macd_line_series': macd_line.tolist(),
                'macd_signal_series': macd_signal.tolist(),
                'macd_histogram_series': macd_histogram.tolist(),
                'macd_crossover': self._get_macd_signal(
                    macd_line.iloc[-1] if not macd_line.empty else 0,
                    macd_signal.iloc[-1] if not macd_signal.empty else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_cci(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Commodity Channel Index"""
        try:
            cci = ta.trend.CCIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.config['cci']['period']
            ).cci()
            
            return {
                'cci': cci.iloc[-1] if not cci.empty else None,
                'cci_series': cci.tolist(),
                'cci_signal': self._get_cci_signal(cci.iloc[-1] if not cci.empty else 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating CCI: {e}")
            return {}
    
    def calculate_obv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate On-Balance Volume"""
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()
            
            # Calculate OBV trend
            obv_sma = obv.rolling(window=10).mean()
            
            return {
                'obv': obv.iloc[-1] if not obv.empty else None,
                'obv_series': obv.tolist(),
                'obv_trend': obv_sma.iloc[-1] if not obv_sma.empty else None,
                'obv_signal': 'bullish' if obv.iloc[-1] > obv_sma.iloc[-1] else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return {}
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        try:
            bollinger = ta.volatility.BollingerBands(
                close=df['close'],
                window=self.config['bollinger']['period'],
                window_dev=self.config['bollinger']['std']
            )
            
            bb_upper = bollinger.bollinger_hband()
            bb_middle = bollinger.bollinger_mavg()
            bb_lower = bollinger.bollinger_lband()
            bb_width = bollinger.bollinger_wband()
            
            current_price = df['close'].iloc[-1]
            
            return {
                'bb_upper': bb_upper.iloc[-1] if not bb_upper.empty else None,
                'bb_middle': bb_middle.iloc[-1] if not bb_middle.empty else None,
                'bb_lower': bb_lower.iloc[-1] if not bb_lower.empty else None,
                'bb_width': bb_width.iloc[-1] if not bb_width.empty else None,
                'bb_upper_series': bb_upper.tolist(),
                'bb_middle_series': bb_middle.tolist(),
                'bb_lower_series': bb_lower.tolist(),
                'bb_position': self._get_bb_position(
                    current_price,
                    bb_upper.iloc[-1] if not bb_upper.empty else None,
                    bb_lower.iloc[-1] if not bb_lower.empty else None
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_psar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Parabolic SAR"""
        try:
            psar = ta.trend.PSARIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                step=self.config['psar']['af'],
                max_step=self.config['psar']['max_af']
            ).psar()
            
            current_price = df['close'].iloc[-1]
            psar_value = psar.iloc[-1] if not psar.empty else None
            
            return {
                'psar': psar_value,
                'psar_series': psar.tolist(),
                'psar_signal': 'bullish' if current_price > psar_value else 'bearish' if psar_value else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error calculating PSAR: {e}")
            return {}
    
    def calculate_signals(self, df: pd.DataFrame, indicators: Dict) -> Dict[str, str]:
        """Calculate overall trading signals based on all indicators"""
        try:
            signals = {}
            current_price = df['close'].iloc[-1]
            
            # VWAP signal
            vwap_value = indicators.get('vwap', {}).get('vwap')
            if vwap_value:
                signals['vwap'] = 'bullish' if current_price > vwap_value else 'bearish'
            
            # RSI signal
            signals['rsi'] = indicators.get('rsi', {}).get('rsi_signal', 'neutral')
            
            # EMA signal
            signals['ema'] = indicators.get('ema', {}).get('ema_alignment', 'neutral')
            
            # MACD signal
            signals['macd'] = indicators.get('macd', {}).get('macd_crossover', 'neutral')
            
            # Overall signal consensus
            bullish_count = sum(1 for signal in signals.values() if signal == 'bullish')
            bearish_count = sum(1 for signal in signals.values() if signal == 'bearish')
            
            if bullish_count > bearish_count:
                signals['overall'] = 'bullish'
            elif bearish_count > bullish_count:
                signals['overall'] = 'bearish'
            else:
                signals['overall'] = 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating signals: {e}")
            return {}
    
    # Helper methods for signal interpretation
    def _get_rsi_signal(self, rsi_value: Optional[float]) -> str:
        if rsi_value is None:
            return 'neutral'
        if rsi_value > self.config['rsi']['overbought']:
            return 'bearish'
        elif rsi_value < self.config['rsi']['oversold']:
            return 'bullish'
        else:
            return 'neutral'
    
    def _get_ema_alignment(self, ema9: Optional[float], ema21: Optional[float], ema50: Optional[float]) -> str:
        if not all([ema9, ema21, ema50]):
            return 'neutral'
        
        if ema9 > ema21 > ema50:
            return 'bullish'
        elif ema9 < ema21 < ema50:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_macd_signal(self, macd: float, signal: float) -> str:
        if macd > signal:
            return 'bullish'
        elif macd < signal:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_cci_signal(self, cci_value: float) -> str:
        if cci_value > self.config['cci']['overbought']:
            return 'bearish'
        elif cci_value < self.config['cci']['oversold']:
            return 'bullish'
        else:
            return 'neutral'
    
    def _get_bb_position(self, price: float, upper: Optional[float], lower: Optional[float]) -> str:
        if not all([upper, lower]):
            return 'neutral'
        
        if price > upper:
            return 'overbought'
        elif price < lower:
            return 'oversold'
        else:
            return 'neutral'
    
    def update_config(self, new_config: Dict):
        """Update indicator configuration"""
        self.config.update(new_config)
        logger.info("Indicator configuration updated")