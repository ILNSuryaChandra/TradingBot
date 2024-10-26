# trading_bot/custom/pattern_analysis.py
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from scipy.signal import argrelextrema
import scipy.stats as stats
from collections import defaultdict
import logging
from .indicators import CustomIndicator, IndicatorResult

class AdvancedPatternRecognition(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.min_pattern_bars = params.get('min_pattern_bars', 5)
        self.max_pattern_bars = params.get('max_pattern_bars', 50)
        self.similarity_threshold = params.get('similarity_threshold', 0.85)
        self.volatility_weight = params.get('volatility_weight', 0.3)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Detect various patterns
            harmonic_patterns = self._find_harmonic_patterns(data)
            elliott_waves = self._analyze_elliott_waves(data)
            wyckoff_phases = self._detect_wyckoff_phases(data)
            volume_patterns = self._analyze_volume_patterns(data)
            
            # Find chart patterns
            chart_patterns = self._find_chart_patterns(data)
            
            # Analyze pattern confluences
            confluence = self._analyze_pattern_confluence(
                harmonic_patterns,
                elliott_waves,
                wyckoff_phases,
                chart_patterns
            )
            
            # Generate signal
            signal = self._generate_signal(confluence)
            strength = self._calculate_pattern_strength(confluence)
            
            return IndicatorResult(
                values=np.array([strength]),
                signal=signal,
                strength=strength,
                additional_data={
                    'harmonic_patterns': harmonic_patterns,
                    'elliott_waves': elliott_waves,
                    'wyckoff_phases': wyckoff_phases,
                    'volume_patterns': volume_patterns,
                    'chart_patterns': chart_patterns,
                    'confluence': confluence
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in AdvancedPatternRecognition: {str(e)}")
            raise
            
    def _find_harmonic_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Detect harmonic patterns (Gartley, Butterfly, Bat, etc.)"""
        patterns = defaultdict(list)
        highs = argrelextrema(data['high'].values, np.greater)[0]
        lows = argrelextrema(data['low'].values, np.less)[0]
        
        def calculate_ratio(move1: float, move2: float) -> float:
            return abs(move2 / move1) if move1 != 0 else 0
            
        for i in range(len(highs) - 4):
            # Get potential XABCD points
            x = data['high'].iloc[highs[i]]
            a = data['low'].iloc[lows[i:i+2][0]]
            b = data['high'].iloc[highs[i+1:i+3][0]]
            c = data['low'].iloc[lows[i+1:i+3][0]]
            d = data['high'].iloc[highs[i+2:i+4][0]]
            
            # Calculate ratios
            ab_bc_ratio = calculate_ratio(b-a, c-b)
            bc_cd_ratio = calculate_ratio(c-b, d-c)
            xab_ratio = calculate_ratio(x-a, b-a)
            
            # Check for Gartley pattern
            if (0.618 <= ab_bc_ratio <= 0.618*1.1 and
                1.27 <= bc_cd_ratio <= 1.618 and
                0.618 <= xab_ratio <= 0.618*1.1):
                patterns['gartley'].append({
                    'type': 'bullish',
                    'points': {'X': x, 'A': a, 'B': b, 'C': c, 'D': d},
                    'confidence': self._calculate_pattern_confidence(
                        [ab_bc_ratio, bc_cd_ratio, xab_ratio],
                        [0.618, 1.618, 0.618]
                    )
                })
                
            # Similar checks for other harmonic patterns...
            
        return dict(patterns)
        
    def _analyze_elliott_waves(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Elliott Wave patterns"""
        waves = {
            'impulse': [],
            'corrective': [],
            'current_wave': None,
            'wave_count': 0
        }
        
        # Find local extrema
        highs = argrelextrema(data['high'].values, np.greater)[0]
        lows = argrelextrema(data['low'].values, np.less)[0]
        
        # Analyze impulse waves (5 waves)
        for i in range(len(highs) - 4):
            wave_points = []
            wave_points.extend(highs[i:i+3])  # Waves 1,3,5
            wave_points.extend(lows[i:i+2])   # Waves 2,4
            wave_points.sort()
            
            if self._validate_impulse_wave(data, wave_points):
                waves['impulse'].append({
                    'start_idx': wave_points[0],
                    'end_idx': wave_points[-1],
                    'points': wave_points,
                    'confidence': self._calculate_wave_confidence(data, wave_points)
                })
                
        # Analyze corrective waves (3 waves)
        for i in range(len(lows) - 2):
            wave_points = []
            wave_points.extend([lows[i], highs[i], lows[i+1]])
            wave_points.sort()
            
            if self._validate_corrective_wave(data, wave_points):
                waves['corrective'].append({
                    'start_idx': wave_points[0],
                    'end_idx': wave_points[-1],
                    'points': wave_points,
                    'confidence': self._calculate_wave_confidence(data, wave_points)
                })
                
        return waves
        
    def _detect_wyckoff_phases(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Wyckoff market phases"""
        phases = {
            'accumulation': [],
            'distribution': [],
            'current_phase': None
        }
        
        volume = data['volume']
        close = data['close']
        
        # Detect Accumulation
        for i in range(50, len(data)):
            window = data.iloc[i-50:i]
            
            # Phase A (Selling Climax)
            if (window['volume'].iloc[-1] > window['volume'].mean() * 2 and
                window['close'].iloc[-1] < window['close'].mean()):
                
                # Phase B (Secondary Test)
                if (window['close'].min() > window['close'].iloc[-1] * 0.98 and
                    window['volume'].iloc[-1] < window['volume'].mean()):
                    
                    # Phase C (Spring)
                    if window['close'].iloc[-1] > window['close'].mean():
                        phases['accumulation'].append({
                            'start_idx': i-50,
                            'end_idx': i,
                            'confidence': self._calculate_wyckoff_confidence(window)
                        })
                        
        # Detect Distribution
        for i in range(50, len(data)):
            window = data.iloc[i-50:i]
            
            # Phase A (Preliminary Supply)
            if (window['volume'].iloc[-1] > window['volume'].mean() * 2 and
                window['close'].iloc[-1] > window['close'].mean()):
                
                # Phase B (Buying Climax)
                if (window['close'].max() < window['close'].iloc[-1] * 1.02 and
                    window['volume'].iloc[-1] < window['volume'].mean()):
                    
                    # Phase C (Upthrust)
                    if window['close'].iloc[-1] < window['close'].mean():
                        phases['distribution'].append({
                            'start_idx': i-50,
                            'end_idx': i,
                            'confidence': self._calculate_wyckoff_confidence(window)
                        })
                        
        return phases
        
    def _analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns and their relationship with price"""
        patterns = {
            'volume_climax': [],
            'volume_divergence': [],
            'volume_confirmation': []
        }
        
        for i in range(20, len(data)):
            window = data.iloc[i-20:i]
            
            # Volume Climax
            if (window['volume'].iloc[-1] > window['volume'].mean() * 2):
                patterns['volume_climax'].append({
                    'index': i,
                    'type': 'buying' if window['close'].iloc[-1] > window['open'].iloc[-1] else 'selling'
                })
                
            # Volume/Price Divergence
            price_trend = window['close'].pct_change().mean()
            volume_trend = window['volume'].pct_change().mean()
            
            if (abs(price_trend) > 0.001 and abs(volume_trend) > 0.001):
                if (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0):
                    patterns['volume_divergence'].append({
                        'index': i,
                        'price_trend': price_trend,
                        'volume_trend': volume_trend
                    })
                    
            # Volume Confirmation
            if (window['close'].iloc[-1] > window['close'].iloc[-2] and
                window['volume'].iloc[-1] > window['volume'].iloc[-2]):
                patterns['volume_confirmation'].append({
                    'index': i,
                    'strength': window['volume'].iloc[-1] / window['volume'].mean()
                })
                
        return patterns
        
    def _find_chart_patterns(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Find classical chart patterns"""
        patterns = {
            'head_and_shoulders': [],
            'double_top': [],
            'double_bottom': [],
            'triangle': [],
            'wedge': []
        }
        
        # Head and Shoulders
        for i in range(50, len(data)):
            window = data.iloc[i-50:i]
            if self._is_head_and_shoulders(window):
                patterns['head_and_shoulders'].append({
                    'start_idx': i-50,
                    'end_idx': i,
                    'confidence': self._calculate_pattern_confidence(window)
                })
                
        # Double Top/Bottom
        for i in range(40, len(data)):
            window = data.iloc[i-40:i]
            top_pattern = self._is_double_top(window)
            bottom_pattern = self._is_double_bottom(window)
            
            if top_pattern:
                patterns['double_top'].append({
                    'start_idx': i-40,
                    'end_idx': i,
                    'confidence': self._calculate_pattern_confidence(window)
                })
            if bottom_pattern:
                patterns['double_bottom'].append({
                    'start_idx': i-40,
                    'end_idx': i,
                    'confidence': self._calculate_pattern_confidence(window)
                })
                
        # Triangle and Wedge patterns
        for i in range(30, len(data)):
            window = data.iloc[i-30:i]
            triangle_type = self._identify_triangle(window)
            wedge_type = self._identify_wedge(window)
            
            if triangle_type:
                patterns['triangle'].append({
                    'start_idx': i-30,
                    'end_idx': i,
                    'type': triangle_type,
                    'confidence': self._calculate_pattern_confidence(window)
                })
            if wedge_type:
                patterns['wedge'].append({
                    'start_idx': i-30,
                    'end_idx': i,
                    'type': wedge_type,
                    'confidence': self._calculate_pattern_confidence(window)
                })
                
        return patterns
        
    def _analyze_pattern_confluence(
        self,
        harmonic_patterns: Dict[str, List[Dict[str, Any]]],
        elliott_waves: Dict[str, Any],
        wyckoff_phases: Dict[str, Any],
        chart_patterns: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze confluence between different pattern types"""
        confluence = {
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'total_confidence': 0,
            'pattern_alignment': []
        }
        
        # Analyze harmonic patterns
        for pattern_type, patterns in harmonic_patterns.items():
            for pattern in patterns:
                if pattern['type'] == 'bullish':
                    confluence['bullish_signals'] += 1
                else:
                    confluence['bearish_signals'] += 1
                confluence['total_confidence'] += pattern['confidence']
                
        # Analyze Elliott waves
        if elliott_waves['current_wave']:
            if elliott_waves['wave_count'] in [1, 3, 5]:
                confluence['bullish_signals'] += 1
            else:
                confluence['bearish_signals'] += 1
                
        # Analyze Wyckoff phases
        if wyckoff_phases['current_phase']:
            if wyckoff_phases['current_phase'] == 'accumulation':
                confluence['bullish_signals'] += 1
            else:
                confluence['bearish_signals'] += 1
                
        # Analyze chart patterns
        for pattern_type, patterns in chart_patterns.items():
            for pattern in patterns:
                if pattern_type in ['double_bottom', 'ascending_triangle']:
                    confluence['bullish_signals'] += 1
                elif pattern_type in ['double_top', 'descending_triangle']:
                    confluence['bearish_signals'] += 1
                else:
                    confluence['neutral_signals'] += 1
                    
        # Calculate overall alignment
        total_signals = (confluence['bullish_signals'] +
                        confluence['bearish_signals'] +
                        confluence['neutral_signals'])
        
        if total_signals > 0:
            confluence['bullish_alignment'] = confluence['bullish_signals'] / total_signals
            confluence['bearish_alignment'] = confluence['bearish_signals'] / total_signals
            
        return confluence
        
    def _generate_signal(self, confluence: Dict[str, Any]) -> str:
        """Generate trading signal based on pattern confluence"""
        if confluence['bullish_alignment'] > 0.6:
            if confluence['total_confidence'] > 0.8:
                return 'strong_buy'
            return 'buy'
        elif confluence['bearish_alignment'] > 0.6:
            if confluence['total_confidence'] > 0.8:
                return 'strong_sell'
            return 'sell'
        return 'neutral'
        
    def _calculate_pattern_strength(self, confluence: Dict[str, Any]) -> float:
        """Calculate overall pattern strength"""
        # Base strength from pattern alignment
        if confluence['bullish_signals'] + confluence['bearish_signals'] == 0:
            return 0.0
            
        # Calculate dominant direction strength
        max_signals = max(confluence['bullish_signals'], confluence['bearish_signals'])
        total_signals = confluence['bullish_signals'] + confluence['bearish_signals'] + confluence['neutral_signals']
        
        # Consider pattern quality
        pattern_quality = confluence['total_confidence'] / (total_signals if total_signals > 0 else 1)
        
        # Calculate final strength
        strength = (max_signals / total_signals) * pattern_quality
        
        return min(max(strength, 0), 1)

    def _detect_double_bottom(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Detect double bottom patterns"""
        result = np.zeros(len(data))
        
        for i in range(window, len(data)):
            section = data['low'].iloc[i-window:i]
            troughs = self._find_troughs(section)
            
            if len(troughs) >= 2:
                last_two_troughs = troughs[-2:]
                if abs(section[last_two_troughs[0]] - section[last_two_troughs[1]]) < section.std() * 0.1:
                    result[i] = 1
                    
        return result
        
    def _detect_head_and_shoulders(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Detect head and shoulders patterns"""
        result = np.zeros(len(data))
        
        for i in range(window, len(data)):
            section = data['high'].iloc[i-window:i]
            peaks = self._find_peaks(section)
            
            if len(peaks) >= 3:
                # Check for head and shoulders pattern
                left = peaks[-3]
                head = peaks[-2]
                right = peaks[-1]
                
                if (section[head] > section[left] and 
                    section[head] > section[right] and
                    abs(section[left] - section[right]) < section.std() * 0.1):
                    result[i] = 1
                    
        return result
        
    def _detect_triangle(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Detect triangle patterns"""
        result = np.zeros(len(data))
        
        for i in range(window, len(data)):
            section = data.iloc[i-window:i]
            highs = section['high']
            lows = section['low']
            
            # Calculate trend lines
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Detect triangle patterns
            if abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
                result[i] = 1  # Symmetrical triangle
            elif high_slope < -0.001 and abs(low_slope) < 0.001:
                result[i] = 2  # Descending triangle
            elif abs(high_slope) < 0.001 and low_slope > 0.001:
                result[i] = 3  # Ascending triangle
                
        return result
        
    def _identify_wedge(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """Identify wedge patterns"""
        result = np.zeros(len(data))
        
        for i in range(window, len(data)):
            section = data.iloc[i-window:i]
            highs = section['high']
            lows = section['low']
            
            # Calculate trend lines
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Detect wedge patterns
            if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
                result[i] = 1  # Falling wedge
            elif high_slope > 0 and low_slope > 0 and high_slope > low_slope:
                result[i] = 2  # Rising wedge
                
        return result