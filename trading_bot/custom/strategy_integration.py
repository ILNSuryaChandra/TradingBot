# trading_bot/custom/strategy_integration.py
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
import logging

from .indicators import (
    CustomIndicator, 
    IndicatorResult,
    MarketRegimeAnalysis,
    VolatilityIndicator,
    MomentumIndicator,
    VolumeProfileIndicator
)
from .ml_enhanced_indicators import MLEnhancedPriceAction
from .pattern_analysis import AdvancedPatternRecognition

@dataclass
class TradeSetup:
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward: float
    confidence: float
    signals: List[str]
    timeframes: List[str]

class AdvancedStrategyIntegration:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators = self._initialize_indicators()
        self.ml_analyzer = MLEnhancedPriceAction(config)
        self.pattern_analyzer = AdvancedPatternRecognition(config)
        
    def _initialize_indicators(self) -> Dict[str, CustomIndicator]:
        """Initialize strategy indicators"""
        try:
            # Directly instantiate each indicator class
            indicators = {
                'market_regime': MarketRegimeAnalysis({
                    'lookback': 100,
                    'volatility_window': 20,
                    'regime_threshold': 0.6
                }),
                'volatility': VolatilityIndicator({
                    'window': 20,
                    'atr_multiplier': 1.5,
                    'std_window': 20
                }),
                'momentum': MomentumIndicator({
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9
                }),
                'volume_profile': VolumeProfileIndicator({
                    'volume_zones': 20,
                    'delta_threshold': 0.6,
                    'imbalance_threshold': 0.7
                })
            }
            
            self.logger.info("Successfully initialized all indicators")
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {str(e)}")
            raise
            
    def analyze_market(
        self,
        data: Dict[str, pd.DataFrame],
        timeframes: List[str]
    ) -> Dict[str, Any]:
        try:
            # Multi-timeframe analysis
            mtf_analysis = self._perform_mtf_analysis(data, timeframes)
            
            # Market regime and context
            market_context = self._analyze_market_context(data, mtf_analysis)
            
            # Advanced pattern recognition
            patterns = self.pattern_analyzer.calculate(data[timeframes[0]])
            
            # ML-enhanced price action
            ml_signals = self.ml_analyzer.calculate(data[timeframes[0]])
            
            # Generate composite signal
            signal = self._generate_composite_signal(
                mtf_analysis,
                market_context,
                patterns,
                ml_signals
            )
            
            # Create trade setup if signal is valid
            trade_setup = self._create_trade_setup(
                signal,
                data[timeframes[0]],
                market_context
            ) if signal['action'] in ['buy', 'sell'] else None
            
            return {
                'signal': signal,
                'trade_setup': trade_setup,
                'market_context': market_context,
                'mtf_analysis': mtf_analysis,
                'patterns': patterns,
                'ml_signals': ml_signals
            }
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            raise
            
    def _perform_mtf_analysis(
        self,
        data: Dict[str, pd.DataFrame],
        timeframes: List[str]
    ) -> Dict[str, Any]:
        """Perform multi-timeframe analysis"""
        try:
            mtf_results = {}
            
            for tf in timeframes:
                tf_data = data[tf]
                
                # Calculate indicators for this timeframe
                regime = self.indicators['market_regime'].calculate(tf_data)
                momentum = self.indicators['momentum'].calculate(tf_data)
                volume = self.indicators['volume_profile'].calculate(tf_data)
                
                mtf_results[tf] = {
                    'regime': regime.signal,
                    'momentum': momentum.signal,
                    'volume': volume.signal,
                    'strength': (regime.strength + momentum.strength + volume.strength) / 3
                }
                
            # Calculate alignment across timeframes
            mtf_results['alignment'] = self._calculate_mtf_alignment(mtf_results)
            
            return mtf_results
            
        except Exception as e:
            self.logger.error(f"Error in MTF analysis: {str(e)}")
            raise
            
    def _analyze_market_context(
        self,
        data: Dict[str, pd.DataFrame],
        mtf_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze overall market context"""
        try:
            # Get primary timeframe data
            primary_tf = list(data.keys())[0]
            primary_data = data[primary_tf]
            
            # Get indicator signals
            regime = self.indicators['market_regime'].calculate(primary_data)
            volatility = self.indicators['volatility'].calculate(primary_data)
            momentum = self.indicators['momentum'].calculate(primary_data)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(regime, volatility, momentum)
            
            return {
                'regime': regime.signal,
                'volatility': volatility.signal,
                'momentum': momentum.signal,
                'risk_score': risk_score,
                'mtf_alignment': mtf_analysis['alignment']
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market context: {str(e)}")
            raise
    
    def _calculate_risk_score(
        self,
        regime: IndicatorResult,
        volatility: IndicatorResult,
        momentum: IndicatorResult
    ) -> float:
        """Calculate overall risk score"""
        try:
            # Weight different components
            volatility_weight = 0.4
            regime_weight = 0.4
            momentum_weight = 0.2
            
            # Calculate weighted risk score
            risk_score = (
                volatility.strength * volatility_weight +
                (1 - regime.strength) * regime_weight +
                abs(momentum.strength) * momentum_weight
            )
            
            return min(max(risk_score, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {str(e)}")
            raise
            
    def _calculate_mtf_alignment(self, mtf_results: Dict[str, Any]) -> float:
        """Calculate alignment across timeframes"""
        try:
            # Collect signals and strengths
            signals = []
            strengths = []
            
            for tf, result in mtf_results.items():
                if tf != 'alignment':  # Skip the alignment key
                    signals.append(result['momentum'])  # Use momentum as primary signal
                    strengths.append(result['strength'])
                    
            # Calculate signal agreement
            bullish = sum(1 for s in signals if s in ['buy', 'strong_buy'])
            bearish = sum(1 for s in signals if s in ['sell', 'strong_sell'])
            
            total_signals = len(signals)
            if total_signals == 0:
                return 0
                
            # Calculate alignment score
            alignment = max(bullish, bearish) / total_signals
            
            # Weight by signal strengths
            weighted_alignment = alignment * np.mean(strengths)
            
            return weighted_alignment
            
        except Exception as e:
            self.logger.error(f"Error calculating MTF alignment: {str(e)}")
            raise

    def _generate_composite_signal(
        self,
        mtf_analysis: Dict[str, Any],
        market_context: Dict[str, Any],
        patterns: IndicatorResult,
        ml_signals: IndicatorResult
    ) -> Dict[str, Any]:
        """Generate composite trading signal"""
        try:
            signal = {
                'action': 'hold',
                'side': None,
                'strength': 0.0,
                'confidence': 0.0,
                'sources': []
            }
            
            # Weight different components
            weights = {
                'mtf': 0.3,
                'market_context': 0.2,
                'patterns': 0.25,
                'ml': 0.25
            }
            
            # Collect and weight signals
            if mtf_analysis['alignment'] > 0.6:
                signal['sources'].append({
                    'type': 'mtf',
                    'strength': mtf_analysis['alignment'] * weights['mtf']
                })
                
            if patterns.signal != 'neutral':
                signal['sources'].append({
                    'type': 'patterns',
                    'strength': patterns.strength * weights['patterns']
                })
                
            if ml_signals.signal != 'neutral':
                signal['sources'].append({
                    'type': 'ml',
                    'strength': ml_signals.strength * weights['ml']
                })
                
            # Calculate final signal
            if signal['sources']:
                total_strength = sum(s['strength'] for s in signal['sources'])
                signal['strength'] = total_strength
                signal['confidence'] = self._calculate_signal_confidence(
                    signal['sources'],
                    market_context
                )
                
                # Determine action and side
                if total_strength > 0.7:
                    signal['action'] = 'enter'
                    signal['side'] = 'buy' if mtf_analysis['alignment'] > 0 else 'sell'
                    
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating composite signal: {str(e)}")
            raise

    def _create_trade_setup(
        self,
        signal: Dict[str, Any],
        data: pd.DataFrame,
        market_context: Dict[str, Any]
    ) -> Optional[TradeSetup]:
        """Create trade setup with entry, exit, and position sizing"""
        try:
            if signal['action'] != 'enter':
                return None
                
            # Calculate entry price
            entry_price = data['close'].iloc[-1]
            
            # Calculate ATR for position sizing
            atr = ta.atr(data['high'], data['low'], data['close']).iloc[-1]
            
            # Calculate stop loss
            stop_distance = atr * self.config['trading']['sl_multiplier']
            stop_loss = (
                entry_price - stop_distance if signal['side'] == 'buy'
                else entry_price + stop_distance
            )
            
            # Calculate take profit
            risk = abs(entry_price - stop_loss)
            reward = risk * self.config['trading']['tp_multiplier']
            take_profit = (
                entry_price + reward if signal['side'] == 'buy'
                else entry_price - reward
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(
                entry_price,
                stop_loss,
                signal['confidence'],
                market_context
            )
            
            return TradeSetup(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_reward=reward / risk if risk != 0 else 0,
                confidence=signal['confidence'],
                signals=[s['type'] for s in signal['sources']],
                timeframes=list(market_context.keys())
            )
            
        except Exception as e:
            self.logger.error(f"Error creating trade setup: {str(e)}")
            return None
            
    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate adaptive position size"""
        try:
            # Get base risk per trade from config
            base_risk = self.config['trading']['position_management']['base_risk_per_trade']
            
            # Adjust risk based on confidence and market conditions
            adjusted_risk = base_risk * confidence * (1 - market_context['risk_score'])
            
            # Calculate position size based on risk amount
            risk_amount = self.config['trading']['position_management']['base_risk_per_trade']
            stop_distance = abs(entry_price - stop_loss)
            
            position_size = (risk_amount * adjusted_risk) / stop_distance
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            raise
            
    def _calculate_signal_confidence(
        self,
        signal_sources: List[Dict[str, Any]],
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate overall signal confidence"""
        try:
            # Base confidence from signal alignment
            base_confidence = np.mean([s['strength'] for s in signal_sources])
            
            # Adjust based on market context
            context_factor = 1.0
            if market_context['regime'] == 'trending':
                context_factor *= 1.2
            elif market_context['regime'] == 'volatile':
                context_factor *= 0.8
                
            # Adjust based on risk score
            risk_factor = 1 - market_context['risk_score']
            
            return min(base_confidence * context_factor * risk_factor, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {str(e)}")
            raise