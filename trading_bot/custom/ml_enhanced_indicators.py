# trading_bot/custom/ml_enhanced_indicators.py
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from dataclasses import dataclass
import logging
from .indicators import CustomIndicator, IndicatorResult

class MLEnhancedPriceAction(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.lookback = params.get('lookback', 100)
        self.prediction_horizon = params.get('prediction_horizon', 5)
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize ML models for different aspects of price action"""
        models = {
            'pattern_recognition': self._create_pattern_recognition_model(),
            'anomaly_detection': IsolationForest(contamination=0.1, random_state=42),
            'trend_prediction': self._create_trend_prediction_model(),
            'support_resistance': DBSCAN(eps=0.3, min_samples=2)
        }
        return models
        
    def _create_pattern_recognition_model(self) -> tf.keras.Model:
        """Create LSTM model for pattern recognition"""
        model = tf.keras.Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback, 5)),
            Dropout(0.2),
            LSTM(30, return_sequences=False),
            Dropout(0.2),
            Dense(3, activation='softmax')  # 3 classes: bullish, bearish, neutral
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def _create_trend_prediction_model(self) -> xgb.XGBRegressor:
        """Create XGBoost model for trend prediction"""
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            objective='reg:squarederror',
            random_state=42
        )
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Prepare features
            features = self._prepare_features(data)
            
            # Pattern recognition
            patterns = self._detect_patterns(features)
            
            # Anomaly detection
            anomalies = self._detect_anomalies(features)
            
            # Trend prediction
            trend_forecast = self._predict_trend(features)
            
            # Support/Resistance levels
            levels = self._find_support_resistance(data)
            
            # Combine signals
            signal = self._generate_signal(patterns, anomalies, trend_forecast, levels)
            strength = self._calculate_strength(patterns, anomalies, trend_forecast)
            
            return IndicatorResult(
                values=trend_forecast,
                signal=signal,
                strength=strength,
                additional_data={
                    'patterns': patterns,
                    'anomalies': anomalies,
                    'trend_forecast': trend_forecast,
                    'support_resistance': levels
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in MLEnhancedPriceAction: {str(e)}")
            raise
            
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        # Price action features
        features.extend([
            data['close'].pct_change(),
            data['high'].pct_change(),
            data['low'].pct_change(),
            data['volume'].pct_change(),
            (data['high'] - data['low']) / data['close']  # Normalized range
        ])
        
        # Technical indicators
        features.extend([
            data.ta.rsi(),
            data.ta.macd()['MACD_12_26_9'],
            data.ta.bbands()['BBU_20_2.0'],
            data.ta.bbands()['BBL_20_2.0']
        ])
        
        features = np.column_stack(features)
        features = self.scaler.fit_transform(features)
        
        return features
        
    def _detect_patterns(self, features: np.ndarray) -> Dict[str, Any]:
        """Detect price patterns using LSTM"""
        # Prepare sequences
        sequences = []
        for i in range(len(features) - self.lookback):
            sequences.append(features[i:i+self.lookback])
        sequences = np.array(sequences)
        
        # Get predictions
        pattern_probs = self.models['pattern_recognition'].predict(sequences)
        
        return {
            'bullish_prob': pattern_probs[-1][0],
            'bearish_prob': pattern_probs[-1][1],
            'neutral_prob': pattern_probs[-1][2]
        }
        
    def _detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect price anomalies using Isolation Forest"""
        return self.models['anomaly_detection'].fit_predict(features)
        
    def _predict_trend(self, features: np.ndarray) -> np.ndarray:
        """Predict future trend using XGBoost"""
        # Prepare lagged features
        X = features[:-self.prediction_horizon]
        y = np.sign(features[self.prediction_horizon:, 0])  # Use price returns as target
        
        # Train and predict
        self.models['trend_prediction'].fit(X, y)
        return self.models['trend_prediction'].predict(features[-self.prediction_horizon:])
        
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels using DBSCAN"""
        # Prepare price levels
        highs = data['high'].values.reshape(-1, 1)
        lows = data['low'].values.reshape(-1, 1)
        
        # Cluster high and low points
        high_clusters = self.models['support_resistance'].fit_predict(highs)
        low_clusters = self.models['support_resistance'].fit_predict(lows)
        
        # Extract levels
        resistance_levels = []
        support_levels = []
        
        for cluster in np.unique(high_clusters)[1:]:  # Skip noise cluster (-1)
            resistance_levels.append(highs[high_clusters == cluster].mean())
            
        for cluster in np.unique(low_clusters)[1:]:
            support_levels.append(lows[low_clusters == cluster].mean())
            
        return {
            'resistance': sorted(resistance_levels),
            'support': sorted(support_levels)
        }

class SmartMoneyAnalysis(CustomIndicator):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.volume_threshold = params.get('volume_threshold', 2.0)
        self.swing_threshold = params.get('swing_threshold', 0.01)
        
    def calculate(self, data: pd.DataFrame) -> IndicatorResult:
        try:
            # Institutional order flow analysis
            order_flow = self._analyze_order_flow(data)
            
            # Liquidity analysis
            liquidity = self._analyze_liquidity(data)
            
            # Price inefficiency analysis
            inefficiencies = self._find_inefficiencies(data)
            
            # Smart money movement patterns
            patterns = self._detect_sm_patterns(data)
            
            # Order block analysis
            order_blocks = self._find_order_blocks(data)
            
            # Generate signal
            signal = self._generate_signal(
                order_flow,
                liquidity,
                inefficiencies,
                patterns,
                order_blocks
            )
            
            strength = self._calculate_strength(
                order_flow,
                liquidity,
                inefficiencies,
                patterns
            )
            
            return IndicatorResult(
                values=order_flow['cumulative_delta'],
                signal=signal,
                strength=strength,
                additional_data={
                    'order_flow': order_flow,
                    'liquidity': liquidity,
                    'inefficiencies': inefficiencies,
                    'patterns': patterns,
                    'order_blocks': order_blocks
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in SmartMoneyAnalysis: {str(e)}")
            raise
            
    def _analyze_order_flow(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze institutional order flow"""
        # Calculate volume delta
        delta = np.where(
            data['close'] > data['open'],
            data['volume'],
            -data['volume']
        )
        
        # Identify large orders
        large_orders = np.where(
            data['volume'] > data['volume'].rolling(20).mean() * self.volume_threshold,
            1,
            0
        )
        
        # Calculate cumulative delta
        cum_delta = pd.Series(delta).cumsum()
        
        # Detect absorption
        absorption = self._detect_absorption(data, delta)
        
        return {
            'delta': delta,
            'cumulative_delta': cum_delta,
            'large_orders': large_orders,
            'absorption': absorption
        }
        
    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market liquidity and institutional levels"""
        # Find high volume nodes
        volume_profile = pd.DataFrame()
        price_levels = np.linspace(data['low'].min(), data['high'].max(), 100)
        
        for i in range(len(data)):
            idx = np.digitize(data['close'].iloc[i], price_levels)
            if idx < len(price_levels):
                if i not in volume_profile.columns:
                    volume_profile[i] = 0
                volume_profile.at[idx, i] = data['volume'].iloc[i]
                
        # Identify liquidity pools
        liquidity_pools = []
        for level in range(len(price_levels)-1):
            if volume_profile.iloc[level].sum() > volume_profile.sum().mean() * self.volume_threshold:
                liquidity_pools.append(price_levels[level])
                
        return {
            'volume_profile': volume_profile,
            'liquidity_pools': liquidity_pools
        }
        
    def _find_inefficiencies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Find price inefficiencies and fair value gaps"""
        inefficiencies = []
        
        for i in range(1, len(data)-1):
            # Look for gaps between candles
            prev_high = data['high'].iloc[i-1]
            curr_low = data['low'].iloc[i]
            curr_high = data['high'].iloc[i]
            next_low = data['low'].iloc[i+1]
            
            # Identify fair value gaps
            if curr_low > prev_high:  # Bullish gap
                inefficiencies.append({
                    'type': 'bullish_gap',
                    'level': (curr_low + prev_high) / 2,
                    'size': curr_low - prev_high
                })
            elif next_low > curr_high:  # Bearish gap
                inefficiencies.append({
                    'type': 'bearish_gap',
                    'level': (next_low + curr_high) / 2,
                    'size': next_low - curr_high
                })
                
        return {
            'gaps': inefficiencies,
            'count': len(inefficiencies)
        }
        
    def _detect_sm_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect smart money movement patterns"""
        patterns = {
            'accumulation': [],
            'distribution': [],
            'manipulation': []
        }
        
        # Detect accumulation
        for i in range(20, len(data)):
            window = data.iloc[i-20:i]
            if (window['volume'].mean() > data['volume'].rolling(50).mean().iloc[i] and
                abs(window['close'].pct_change().mean()) < self.swing_threshold):
                patterns['accumulation'].append(i)
                
        # Detect distribution
        for i in range(20, len(data)):
            window = data.iloc[i-20:i]
            if (window['volume'].mean() > data['volume'].rolling(50).mean().iloc[i] and
                window['close'].std() > data['close'].rolling(50).std().iloc[i]):
                patterns['distribution'].append(i)
                
        # Detect manipulation
        for i in range(5, len(data)):
            if (data['high'].iloc[i] > data['high'].iloc[i-5:i].max() and
                data['close'].iloc[i] < data['open'].iloc[i]):
                patterns['manipulation'].append(i)
                
        return patterns
        
    def _find_order_blocks(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Find institutional order blocks"""
        order_blocks = {
            'bullish': [],
            'bearish': []
        }
        
        for i in range(1, len(data)-1):
            # Bullish order block
            if (data['close'].iloc[i] < data['open'].iloc[i] and  # Bearish candle
                data['volume'].iloc[i] > data['volume'].rolling(20).mean().iloc[i] and
                data['low'].iloc[i+1:i+5].min() > data['close'].iloc[i]):
                
                order_blocks['bullish'].append({
                    'index': i,
                    'top': data['open'].iloc[i],
                    'bottom': data['close'].iloc[i],
                    'volume': data['volume'].iloc[i]
                })
                
            # Bearish order block
            if (data['close'].iloc[i] > data['open'].iloc[i] and  # Bullish candle
                data['volume'].iloc[i] > data['volume'].rolling(20).mean().iloc[i] and
                data['high'].iloc[i+1:i+5].max() < data['close'].iloc[i]):
                
                order_blocks['bearish'].append({
                    'index': i,
                    'top': data['close'].iloc[i],
                    'bottom': data['open'].iloc[i],
                    'volume': data['volume'].iloc[i]
                })
                
        return order_blocks
        
    def _detect_absorption(
        self,
        data: pd.DataFrame,
        delta: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect volume absorption"""
        absorption_points = []
        
        for i in range(20, len(data)):
            window_delta = delta[i-20:i]
            window_volume = data['volume'].iloc[i-20:i]
            
            # Check for absorption conditions
            if (abs(window_delta.sum()) < window_volume.mean() * 0.2 and
                data['volume'].iloc[i] > window_volume.mean() * 1.5):
                
                absorption_points.append({
                    'index': i,
                    'price': data['close'].iloc[i],
                    'volume': data['volume'].iloc[i]
                })
            
            return absorption_points