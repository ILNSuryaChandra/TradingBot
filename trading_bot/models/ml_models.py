# trading_bot/models/ml_models.py
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import joblib
import logging
from datetime import datetime
from pathlib import Path

class FeatureEngineering:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.custom_features = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Create copy to avoid modifying original data
            data = df.copy()
            
            # Add basic technical indicators using pandas_ta
            data.ta.strategy(
                "AllStrategy",
                verbose=False
            )
            
            # Add custom momentum features
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log1p(data['returns'])
            
            # Add volatility features
            for window in [5, 10, 20]:
                data[f'volatility_{window}'] = data['returns'].rolling(window).std()
                
            # Add price patterns
            data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
            data['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)
            
            # Volume analysis
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
            
            # Drop rows with NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            raise
            
    def add_custom_feature(self, name: str, calculation_func: callable):
        """Add a custom feature calculation"""
        self.custom_features[name] = calculation_func

class ModelEnsemble:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.sequence_length = 10
        self.feature_dims = None
        self._initialize_models()
        
    def _initialize_models(self):
        try:
            # Initialize XGBoost model
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
            
            # LSTM will be initialized during first training when we know input dimensions
            self.models['lstm'] = None
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
            
    def _initialize_lstm(self, input_shape: Tuple[int, int]):
        """Initialize LSTM model with proper input shape"""
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(30, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error initializing LSTM model: {str(e)}")
            raise
            
    def prepare_data(
        self,
        df: pd.DataFrame,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            if sequence_length:
                self.sequence_length = sequence_length
                
            # Create features
            data = df.copy()
            
            # Add technical indicators
            data.ta.strategy("AllStrategy")
            
            # Remove non-numeric columns and NaN values
            features = data.select_dtypes(include=[np.number]).fillna(0)
            
            # Remove target column if exists
            if 'close' in features.columns:
                target = features['close'].pct_change().shift(-1)  # Next period returns
                features = features.drop(columns=['close'])
            else:
                target = pd.Series(np.zeros(len(features)))
            
            # Scale features
            if 'features' not in self.scalers:
                self.scalers['features'] = StandardScaler()
                scaled_features = self.scalers['features'].fit_transform(features)
            else:
                scaled_features = self.scalers['features'].transform(features)
            
            # Create sequences for LSTM
            X_lstm, y = [], []
            for i in range(len(scaled_features) - self.sequence_length):
                X_lstm.append(scaled_features[i:i + self.sequence_length])
                y.append(target.iloc[i + self.sequence_length])
            
            X_lstm = np.array(X_lstm)
            X_xgb = scaled_features[self.sequence_length:]
            y = np.array(y)
            
            # Initialize LSTM if not done yet
            if self.models['lstm'] is None:
                self.feature_dims = X_lstm.shape[2]
                self.models['lstm'] = self._initialize_lstm((self.sequence_length, self.feature_dims))
            
            return X_lstm, X_xgb, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def train(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2
    ):
        try:
            # Prepare data
            X_lstm, X_xgb, y = self.prepare_data(df)
            
            # Split data
            split_idx = int(len(X_lstm) * (1 - validation_split))
            
            X_lstm_train = X_lstm[:split_idx]
            X_lstm_val = X_lstm[split_idx:]
            X_xgb_train = X_xgb[:split_idx]
            X_xgb_val = X_xgb[split_idx:]
            y_train = y[:split_idx]
            y_val = y[split_idx:]
            
            # Train XGBoost
            self.models['xgboost'].fit(
                X_xgb_train,
                y_train,
                eval_set=[(X_xgb_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Train LSTM
            self.models['lstm'].fit(
                X_lstm_train,
                y_train,
                validation_data=(X_lstm_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            self.logger.info("Models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise
            
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        try:
            # Prepare prediction data
            X_lstm, X_xgb, _ = self.prepare_data(df)
            
            # Get predictions from both models
            xgb_pred = self.models['xgboost'].predict(X_xgb[-1:])
            lstm_pred = self.models['lstm'].predict(X_lstm[-1:])
            
            # Ensemble prediction (weighted average)
            ensemble_pred = 0.6 * xgb_pred[-1] + 0.4 * lstm_pred[-1][0]
            
            return {
                'xgboost': float(xgb_pred[-1]),
                'lstm': float(lstm_pred[-1][0]),
                'ensemble': float(ensemble_pred)
            }
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def save_models(self, directory: str):
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            
            # Save XGBoost
            joblib.dump(self.models['xgboost'], path / 'xgboost_model.joblib')
            
            # Save LSTM
            if self.models['lstm'] is not None:
                self.models['lstm'].save(str(path / 'lstm_model.h5'))
            
            # Save scalers
            joblib.dump(self.scalers, path / 'scalers.joblib')
            
            self.logger.info(f"Models saved to {directory}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
            
    def load_models(self, directory: str):
        try:
            path = Path(directory)
            
            # Load XGBoost
            if (path / 'xgboost_model.joblib').exists():
                self.models['xgboost'] = joblib.load(path / 'xgboost_model.joblib')
            
            # Load LSTM
            if (path / 'lstm_model.h5').exists():
                self.models['lstm'] = load_model(str(path / 'lstm_model.h5'))
            
            # Load scalers
            if (path / 'scalers.joblib').exists():
                self.scalers = joblib.load(path / 'scalers.joblib')
            
            self.logger.info(f"Models loaded from {directory}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise

class TradingStrategy:
    def __init__(
        self,
        config: Dict[str, Any],
        model_ensemble: ModelEnsemble
    ):
        self.config = config
        self.model_ensemble = model_ensemble
        self.logger = logging.getLogger(__name__)
        
    def analyze_market(
        self,
        df: pd.DataFrame,
        threshold: float = 0.001
    ) -> Dict[str, Any]:
        try:
            # Get model predictions
            predictions = self.model_ensemble.predict(df)
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(df)
            
            # Generate trading signal
            signal = self._generate_signal(
                predictions['ensemble'],
                market_conditions,
                threshold
            )
            
            return {
                'signal': signal,
                'prediction': predictions['ensemble'],
                'confidence': self._calculate_confidence(predictions, market_conditions),
                'market_conditions': market_conditions
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market: {str(e)}")
            raise
            
    def _analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            data = df.copy()
            
            # Calculate market regime indicators
            volatility = data['close'].pct_change().std()
            trend = data.ta.ema(length=20) - data.ta.ema(length=50)
            rsi = data.ta.rsi()
            
            # Determine market regime
            if volatility > data['close'].pct_change().std().rolling(20).mean().iloc[-1] * 1.5:
                regime = "volatile"
            elif abs(trend.iloc[-1]) > trend.std() * 1.5:
                regime = "trending"
            elif rsi.iloc[-1] > 70 or rsi.iloc[-1] < 30:
                regime = "ranging"
            else:
                regime = "stable"
                
            return {
                'regime': regime,
                'volatility': float(volatility),
                'trend_strength': float(abs(trend.iloc[-1])),
                'rsi': float(rsi.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            raise
            
    def _generate_signal(
        self,
        prediction: float,
        market_conditions: Dict[str, Any],
        threshold: float
    ) -> Dict[str, Any]:
        try:
            signal = {
                'action': 'hold',
                'side': None,
                'strength': 0.0
            }
            
            # Adjust threshold based on market conditions
            adjusted_threshold = self._adjust_threshold(
                threshold,
                market_conditions
            )
            
            # Generate signal based on prediction and adjusted threshold
            if prediction > adjusted_threshold:
                signal['action'] = 'enter'
                signal['side'] = 'buy'
                signal['strength'] = prediction / adjusted_threshold
            elif prediction < -adjusted_threshold:
                signal['action'] = 'enter'
                signal['side'] = 'sell'
                signal['strength'] = abs(prediction / adjusted_threshold)
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            raise
            
    def _adjust_threshold(
        self,
        base_threshold: float,
        market_conditions: Dict[str, Any]
    ) -> float:
        try:
            # Adjust threshold based on market regime
            regime_multipliers = {
                'volatile': 1.5,
                'trending': 0.8,
                'ranging': 1.2,
                'stable': 1.0
            }
            
            regime_multiplier = regime_multipliers[market_conditions['regime']]
            volatility_adjustment = market_conditions['volatility'] / 0.02  # baseline volatility
            
            return base_threshold * regime_multiplier * volatility_adjustment
            
        except Exception as e:
            self.logger.error(f"Error adjusting threshold: {str(e)}")
            raise
            
    def _calculate_confidence(
        self,
        predictions: Dict[str, float],
        market_conditions: Dict[str, Any]
    ) -> float:
        try:
            # Calculate model agreement
            pred_values = [predictions['xgboost'], predictions['lstm']]
            model_agreement = 1 - (np.std(pred_values) / abs(np.mean(pred_values)))
            
            # Calculate market confidence
            market_confidence = self._calculate_market_confidence(market_conditions)
            
            # Combine confidences
            confidence = (0.6 * model_agreement + 0.4 * market_confidence)
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            raise
            
    def _calculate_market_confidence(
        self,
        market_conditions: Dict[str, Any]
    ) -> float:
        try:
            # Define ideal conditions for trading
            ideal_conditions = {
                'volatile': {'volatility': 0.02, 'trend_strength': 0.02},
                'trending': {'volatility': 0.015, 'trend_strength': 0.03},
                'ranging': {'volatility': 0.01, 'trend_strength': 0.01},
                'stable': {'volatility': 0.005, 'trend_strength': 0.015}
            }
            
            # Calculate deviation from ideal conditions
            ideal = ideal_conditions[market_conditions['regime']]
            volatility_diff = abs(market_conditions['volatility'] - ideal['volatility'])
            trend_diff = abs(market_conditions['trend_strength'] - ideal['trend_strength'])
            
            # Convert differences to confidence scores
            volatility_confidence = 1 / (1 + volatility_diff * 50)
            trend_confidence = 1 / (1 + trend_diff * 50)
            
            return float(np.mean([volatility_confidence, trend_confidence]))
            
        except Exception as e:
            self.logger.error(f"Error calculating market confidence: {str(e)}")
            raise
