# trading_bot/core/state_manager.py

from typing import Dict, List, Optional, Union, Any
import json
import pickle
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
from dataclasses import asdict, dataclass
from .base import TradePosition, OrderData

@dataclass
class TradingState:
    active_positions: Dict[str, TradePosition]
    pending_orders: Dict[str, OrderData]
    last_model_update: datetime
    last_trade_time: Dict[str, datetime]
    account_state: Dict[str, Any]
    market_state: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class StateManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state_dir = Path(config.get('state_path', 'state'))
        self.state_file = self.state_dir / 'trading_state.pkl'
        self.backup_dir = self.state_dir / 'backups'
        self.initialize_state_directory()

    def initialize_state_directory(self):
        """Create necessary directories for state management"""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating state directories: {str(e)}")
            raise

    def save_state(self, trading_state: TradingState):
        """Save current trading state to disk"""
        try:
            # Create backup of current state if it exists
            if self.state_file.exists():
                self._create_backup()

            # Convert datetime objects to ISO format for JSON serialization
            state_dict = self._prepare_state_for_saving(trading_state)

            # Save state
            with open(self.state_file, 'wb') as f:
                pickle.dump(state_dict, f)

            self.logger.info("Trading state saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving trading state: {str(e)}")
            raise

    def load_state(self) -> Optional[TradingState]:
        """Load trading state from disk"""
        try:
            if not self.state_file.exists():
                self.logger.info("No existing state found")
                return None

            with open(self.state_file, 'rb') as f:
                state_dict = pickle.load(f)

            # Convert ISO format strings back to datetime objects
            restored_state = self._restore_state_from_dict(state_dict)
            self.logger.info("Trading state loaded successfully")
            return restored_state

        except Exception as e:
            self.logger.error(f"Error loading trading state: {str(e)}")
            self._attempt_state_recovery()
            raise

    def _create_backup(self):
        """Create a backup of the current state file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f'trading_state_{timestamp}.pkl'
            
            with open(self.state_file, 'rb') as src, open(backup_file, 'wb') as dst:
                dst.write(src.read())

        except Exception as e:
            self.logger.error(f"Error creating state backup: {str(e)}")
            raise

    def _attempt_state_recovery(self):
        """Attempt to recover state from most recent backup"""
        try:
            backup_files = sorted(self.backup_dir.glob('trading_state_*.pkl'))
            if not backup_files:
                self.logger.error("No backup files found for recovery")
                return None

            latest_backup = backup_files[-1]
            self.logger.info(f"Attempting recovery from backup: {latest_backup}")

            with open(latest_backup, 'rb') as f:
                state_dict = pickle.load(f)

            return self._restore_state_from_dict(state_dict)

        except Exception as e:
            self.logger.error(f"Error recovering state from backup: {str(e)}")
            raise

    def _prepare_state_for_saving(self, trading_state: TradingState) -> Dict[str, Any]:
        """Prepare state for serialization"""
        state_dict = {
            'active_positions': {
                symbol: asdict(pos) for symbol, pos in trading_state.active_positions.items()
            },
            'pending_orders': {
                order_id: asdict(order) for order_id, order in trading_state.pending_orders.items()
            },
            'last_model_update': trading_state.last_model_update.isoformat(),
            'last_trade_time': {
                symbol: dt.isoformat() for symbol, dt in trading_state.last_trade_time.items()
            },
            'account_state': trading_state.account_state,
            'market_state': trading_state.market_state,
            'performance_metrics': trading_state.performance_metrics
        }
        return state_dict

    def _restore_state_from_dict(self, state_dict: Dict[str, Any]) -> TradingState:
        """Restore trading state from dictionary"""
        active_positions = {
            symbol: TradePosition(**pos_dict)
            for symbol, pos_dict in state_dict['active_positions'].items()
        }

        pending_orders = {
            order_id: OrderData(**order_dict)
            for order_id, order_dict in state_dict['pending_orders'].items()
        }

        last_model_update = datetime.fromisoformat(state_dict['last_model_update'])
        last_trade_time = {
            symbol: datetime.fromisoformat(dt_str)
            for symbol, dt_str in state_dict['last_trade_time'].items()
        }

        return TradingState(
            active_positions=active_positions,
            pending_orders=pending_orders,
            last_model_update=last_model_update,
            last_trade_time=last_trade_time,
            account_state=state_dict['account_state'],
            market_state=state_dict['market_state'],
            performance_metrics=state_dict['performance_metrics']
        )

    def clear_state(self):
        """Clear all state data"""
        try:
            if self.state_file.exists():
                self._create_backup()
                self.state_file.unlink()
            self.logger.info("Trading state cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing trading state: {str(e)}")
            raise
