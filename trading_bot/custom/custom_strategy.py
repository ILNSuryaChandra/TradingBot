# trading_bot/custom/strategy.py
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
import pandas as pd
import pandas_ta as ta
from abc import ABC, abstractmethod
import logging
from .indicators import CustomIndicator, IndicatorResult
from .custom_indicators import CustomVolumeProfile, CustomMarketStructure, CustomMomentumIndex

class BaseStrategy(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.indicators: Dict[str, CustomIndicator] = {}
        
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass
        
    def add_indicator(self, name: str, indicator: CustomIndicator):
        self.indicators[name] = indicator
        
    def remove_indicator(self, name: str):
        if name in self.indicators:
            del self.indicators[name]

class StrategyBuilder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.entry_conditions: List[Callable] = []
        self.exit_conditions: List[Callable] = []
        self.risk_rules: List[Callable] = []
        self.indicators: Dict[str, CustomIndicator] = {}
        
    def add_entry_condition(self, condition: Callable):
        self.entry_conditions.append(condition)
        return self
        
    def add_exit_condition(self, condition: Callable):
        self.exit_conditions.append(condition)
        return self
        
    def add_risk_rule(self, rule: Callable):
        self.risk_rules.append(rule)
        return self
        
    def add_indicator(self, name: str, indicator: CustomIndicator):
        self.indicators[name] = indicator
        return self
        
    def build(self) -> 'CustomStrategy':
        return CustomStrategy(
            entry_conditions=self.entry_conditions,
            exit_conditions=self.exit_conditions,
            risk_rules=self.risk_rules,
            indicators=self.indicators
        )

class CustomStrategy(BaseStrategy):
    def __init__(
        self,
        entry_conditions: List[Callable],
        exit_conditions: List[Callable],
        risk_rules: List[Callable],
        indicators: Dict[str, CustomIndicator]
    ):
        self.entry_conditions = entry_conditions
        self.exit_conditions = exit_conditions
        self.risk_rules = risk_rules
        self.indicators = indicators
        self.logger = logging.getLogger(__name__)
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            # Calculate indicator values
            indicator_results = self._calculate_indicators(data)
            
            # Check entry conditions
            entry_signals = self._check_entry_conditions(data, indicator_results)
            
            # Check exit conditions
            exit_signals = self._check_exit_conditions(data, indicator_results)
            
            # Apply risk rules
            risk_assessment = self._apply_risk_rules(data, indicator_results)
            
            # Generate final signal
            signal = self._combine_signals(
                entry_signals,
                exit_signals,
                risk_assessment
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            raise
            
    def _calculate_indicators(
        self,
        data: pd.DataFrame
    ) -> Dict[str, IndicatorResult]:
        results = {}
        for name, indicator in self.indicators.items():
            results[name] = indicator.calculate(data)
        return results
        
    def _check_entry_conditions(
        self,
        data: pd.DataFrame,
        indicator_results: Dict[str, IndicatorResult]
    ) -> List[Dict[str, Any]]:
        signals = []
        for condition in self.entry_conditions:
            try:
                signal = condition(data, indicator_results)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error in entry condition: {str(e)}")
        return signals
        
    def _check_exit_conditions(
        self,
        data: pd.DataFrame,
        indicator_results: Dict[str, IndicatorResult]
    ) -> List[Dict[str, Any]]:
        signals = []
        for condition in self.exit_conditions:
            try:
                signal = condition(data, indicator_results)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error in exit condition: {str(e)}")
        return signals
        
    def _apply_risk_rules(
        self,
        data: pd.DataFrame,
        indicator_results: Dict[str, IndicatorResult]
    ) -> Dict[str, Any]:
        risk_assessment = {
            'allow_trade': True,
            'risk_score': 1.0,
            'warnings': []
        }
        
        for rule in self.risk_rules:
            try:
                result = rule(data, indicator_results)
                if not result['allow_trade']:
                    risk_assessment['allow_trade'] = False
                risk_assessment['risk_score'] *= result.get('risk_score', 1.0)
                risk_assessment['warnings'].extend(result.get('warnings', []))
            except Exception as e:
                self.logger.error(f"Error in risk rule: {str(e)}")
                
        return risk_assessment
        
    def _combine_signals(
        self,
        entry_signals: List[Dict[str, Any]],
        exit_signals: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not risk_assessment['allow_trade']:
            return {
                'action': 'hold',
                'side': None,
                'strength': 0.0,
                'reason': 'Risk rules violation',
                'warnings': risk_assessment['warnings']
            }
            
        if exit_signals:
            return {
                'action': 'exit',
                'side': None,
                'strength': max(s.get('strength', 0.0) for s in exit_signals),
                'reason': 'Exit conditions met',
                'signals': exit_signals
            }
            
        if entry_signals:
            # Combine entry signals
            long_signals = [s for s in entry_signals if s.get('side') == 'buy']
            short_signals = [s for s in entry_signals if s.get('side') == 'sell']
            
            if long_signals and not short_signals:
                return {
                    'action': 'enter',
                    'side': 'buy',
                    'strength': sum(s.get('strength', 0.0) for s in long_signals),
                    'reason': 'Long entry conditions met',
                    'signals': long_signals,
                    'risk_score': risk_assessment['risk_score']
                }
            elif short_signals and not long_signals:
                return {
                    'action': 'enter',
                    'side': 'sell',
                    'strength': sum(s.get('strength', 0.0) for s in short_signals),
                    'reason': 'Short entry conditions met',
                    'signals': short_signals,
                    'risk_score': risk_assessment['risk_score']
                }
                
        return {
            'action': 'hold',
            'side': None,
            'strength': 0.0,
            'reason': 'No clear signals',
            'risk_score': risk_assessment['risk_score']
        }

# Example strategy implementation
class VolumeBreakoutStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.setup_indicators()
        
    def setup_indicators(self):
        # Add custom indicators
        self.add_indicator('volume_profile', CustomVolumeProfile({
            'volume_zones': 20,
            'value_area_volume': 0.68
        }))
        
        self.add_indicator('market_structure', CustomMarketStructure({
            'window': 5
        }))
        
        self.add_indicator('momentum', CustomMomentumIndex({
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }))
        
    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            # Calculate indicator values
            volume_profile = self.indicators['volume_profile'].calculate(data)
            market_structure = self.indicators['market_structure'].calculate(data)
            momentum = self.indicators['momentum'].calculate(data)
            
            current_price = data['close'].iloc[-1]
            current_volume = data['volume'].iloc[-1]
            
            # Check for volume breakout conditions
            volume_breakout = current_volume > data['volume'].rolling(20).mean().iloc[-1] * 1.5
            
            # Generate signal based on conditions
            signal = {
                'action': 'hold',
                'side': None,
                'strength': 0.0,
                'reason': 'No clear signal'
            }
            
            if volume_breakout:
                # Check if price is near volume profile POC
                poc_price = volume_profile.additional_data['poc_price']
                price_near_poc = abs(current_price - poc_price) / poc_price < 0.002
                
                if price_near_poc and market_structure.signal == 'uptrend':
                    signal = {
                        'action': 'enter',
                        'side': 'buy',
                        'strength': momentum.strength,
                        'reason': 'Volume breakout with uptrend'
                    }
                elif price_near_poc and market_structure.signal == 'downtrend':
                    signal = {
                        'action': 'enter',
                        'side': 'sell',
                        'strength': momentum.strength,
                        'reason': 'Volume breakout with downtrend'
                    }
                    
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            raise

# Example usage
def create_custom_strategy(config: Dict[str, Any]) -> CustomStrategy:
    builder = StrategyBuilder()
    
    # Add entry conditions
    builder.add_entry_condition(
        lambda data, indicators: {
            'side': 'buy',
            'strength': 1.0
        } if indicators['momentum'].signal == 'strong_buy' else None
    )
    
    # Add exit conditions
    builder.add_exit_condition(
        lambda data, indicators: {
            'side': 'exit',
            'strength': 1.0
        } if indicators['momentum'].signal == 'strong_sell' else None
    )
    
    # Add risk rules
    builder.add_risk_rule(
        lambda data, indicators: {
            'allow_trade': indicators['market_structure'].strength > 0.5,
            'risk_score': indicators['market_structure'].strength,
            'warnings': ['Low market structure strength']
            if indicators['market_structure'].strength <= 0.5 else []
        }
    )
    
    # Add indicators
    builder.add_indicator('momentum', CustomMomentumIndex({
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }))
    
    builder.add_indicator('market_structure', CustomMarketStructure({
        'window': 5
    }))
    
    return builder.build()
