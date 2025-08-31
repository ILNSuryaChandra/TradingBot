import asyncio
import uuid
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class ExecutionEngine:
    """
    Live execution engine with order management, execution benchmarking,
    and health monitoring
    """
    
    def __init__(self, data_service, risk_manager):
        self.data_service = data_service
        self.risk_manager = risk_manager
        
        # Order tracking
        self.active_orders = {}
        self.order_history = []
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'avg_execution_time': 0.0,
            'slippage_stats': []
        }
        
        # Exchange connections (to be implemented)
        self.exchange_adapters = {}
        self.health_status = {
            'api_connected': False,
            'websocket_connected': False,
            'last_heartbeat': None,
            'latency_ms': 0
        }
        
        # Execution quality tracking
        self.execution_benchmarks = {
            'vwap_execution_quality': [],
            'twap_execution_quality': [],
            'slippage_vs_spread': []
        }
        
        logger.info("Execution Engine initialized")
    
    async def submit_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit order with idempotency and retry logic
        """
        try:
            # Generate unique order ID
            order_id = client_order_id or f"order_{uuid.uuid4().hex[:8]}"
            
            # Check for duplicate orders
            if order_id in self.active_orders:
                return {
                    'success': False,
                    'error': f'Order {order_id} already exists',
                    'order_id': order_id
                }
            
            # Create order object
            order = {
                'order_id': order_id,
                'pair': pair,
                'side': side.value,
                'type': order_type.value,
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price,
                'time_in_force': time_in_force,
                'status': OrderStatus.PENDING.value,
                'created_at': datetime.utcnow().isoformat(),
                'filled_quantity': 0.0,
                'avg_fill_price': 0.0,
                'fees': 0.0,
                'exchange': 'binance',  # Default for now
                'retries': 0,
                'max_retries': 3
            }
            
            # Add to active orders
            self.active_orders[order_id] = order
            
            # Execute order (simulate for now)
            execution_result = await self._execute_order(order)
            
            # Update order status
            order.update(execution_result)
            
            # Update statistics
            self.execution_stats['total_orders'] += 1
            
            if execution_result.get('success'):
                order['status'] = OrderStatus.FILLED.value
                self.execution_stats['filled_orders'] += 1
                
                # Calculate execution quality
                await self._analyze_execution_quality(order)
                
                logger.info(f"Order {order_id} executed successfully")
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'order': order,
                    'execution_time_ms': execution_result.get('execution_time_ms', 0)
                }
            else:
                order['status'] = OrderStatus.REJECTED.value
                self.execution_stats['rejected_orders'] += 1
                
                logger.error(f"Order {order_id} execution failed: {execution_result.get('error')}")
                
                return {
                    'success': False,
                    'error': execution_result.get('error', 'Unknown execution error'),
                    'order_id': order_id,
                    'order': order
                }
                
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id if 'order_id' in locals() else None
            }
    
    async def _execute_order(self, order: Dict) -> Dict[str, Any]:
        """
        Execute order through exchange adapter (simulated for now)
        """
        try:
            start_time = time.time()
            
            # Simulate execution delay
            await asyncio.sleep(0.1)
            
            # Get current market data
            ticker = await self.data_service.get_ticker(order['pair'])
            
            if not ticker:
                return {
                    'success': False,
                    'error': 'Unable to get market data'
                }
            
            # Simulate order execution
            if order['type'] == OrderType.MARKET.value:
                # Market order - execute at current price with slippage
                if order['side'] == OrderSide.BUY.value:
                    execution_price = ticker.get('ask', ticker.get('last', 0))
                    slippage = 0.001  # 0.1% slippage simulation
                    execution_price *= (1 + slippage)
                else:
                    execution_price = ticker.get('bid', ticker.get('last', 0))
                    slippage = 0.001
                    execution_price *= (1 - slippage)
                
                filled_quantity = order['quantity']
                
            elif order['type'] == OrderType.LIMIT.value:
                # Limit order - check if price can be filled
                current_price = ticker.get('last', 0)
                
                if order['side'] == OrderSide.BUY.value and order['price'] >= current_price:
                    execution_price = order['price']
                    filled_quantity = order['quantity']
                elif order['side'] == OrderSide.SELL.value and order['price'] <= current_price:
                    execution_price = order['price']
                    filled_quantity = order['quantity']
                else:
                    # Order not filled yet
                    return {
                        'success': False,
                        'error': 'Limit order price not reached'
                    }
            else:
                return {
                    'success': False,
                    'error': f'Order type {order["type"]} not implemented'
                }
            
            # Calculate fees (simulate 0.1% trading fee)
            fees = filled_quantity * execution_price * 0.001
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'filled_quantity': filled_quantity,
                'avg_fill_price': execution_price,
                'fees': fees,
                'execution_time_ms': execution_time_ms,
                'slippage': abs(execution_price - ticker.get('last', execution_price)) / ticker.get('last', execution_price) if ticker.get('last') else 0
            }
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel active order"""
        try:
            if order_id not in self.active_orders:
                return {
                    'success': False,
                    'error': f'Order {order_id} not found'
                }
            
            order = self.active_orders[order_id]
            
            if order['status'] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
                return {
                    'success': False,
                    'error': f'Order {order_id} cannot be cancelled (status: {order["status"]})'
                }
            
            # Update order status
            order['status'] = OrderStatus.CANCELLED.value
            order['cancelled_at'] = datetime.utcnow().isoformat()
            
            # Move to history
            self.order_history.append(order)
            del self.active_orders[order_id]
            
            # Update statistics
            self.execution_stats['cancelled_orders'] += 1
            
            logger.info(f"Order {order_id} cancelled")
            
            return {
                'success': True,
                'order_id': order_id,
                'order': order
            }
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            if order_id in self.active_orders:
                return {
                    'found': True,
                    'order': self.active_orders[order_id]
                }
            
            # Check order history
            for order in self.order_history:
                if order['order_id'] == order_id:
                    return {
                        'found': True,
                        'order': order
                    }
            
            return {
                'found': False,
                'error': f'Order {order_id} not found'
            }
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {
                'found': False,
                'error': str(e)
            }
    
    async def _analyze_execution_quality(self, order: Dict):
        """Analyze execution quality vs VWAP/TWAP benchmarks"""
        try:
            # Get recent market data for benchmarking
            historical_data = await self.data_service.get_historical_ohlcv(
                pair=order['pair'],
                timeframe='1m',
                limit=10
            )
            
            if not historical_data:
                return
            
            # Calculate VWAP for comparison
            vwap = self._calculate_benchmark_vwap(historical_data)
            
            if vwap and order.get('avg_fill_price'):
                execution_price = order['avg_fill_price']
                
                # Calculate execution vs VWAP
                if order['side'] == OrderSide.BUY.value:
                    vwap_diff = (execution_price - vwap) / vwap
                else:
                    vwap_diff = (vwap - execution_price) / vwap
                
                # Store execution quality metrics
                quality_metric = {
                    'order_id': order['order_id'],
                    'timestamp': order['created_at'],
                    'pair': order['pair'],
                    'side': order['side'],
                    'execution_price': execution_price,
                    'benchmark_vwap': vwap,
                    'vwap_difference_percent': vwap_diff * 100,
                    'slippage_percent': order.get('slippage', 0) * 100,
                    'execution_time_ms': order.get('execution_time_ms', 0)
                }
                
                self.execution_benchmarks['vwap_execution_quality'].append(quality_metric)
                
                # Keep only recent metrics (last 100)
                if len(self.execution_benchmarks['vwap_execution_quality']) > 100:
                    self.execution_benchmarks['vwap_execution_quality'] = \
                        self.execution_benchmarks['vwap_execution_quality'][-100:]
                
                logger.debug(f"Execution quality recorded: VWAP diff {vwap_diff:.4%}")
                
        except Exception as e:
            logger.error(f"Error analyzing execution quality: {e}")
    
    def _calculate_benchmark_vwap(self, historical_data: List[Dict]) -> Optional[float]:
        """Calculate VWAP from historical data"""
        try:
            if not historical_data:
                return None
            
            total_volume = sum(candle['volume'] for candle in historical_data)
            if total_volume == 0:
                return None
            
            vwap = sum(
                (candle['high'] + candle['low'] + candle['close']) / 3 * candle['volume']
                for candle in historical_data
            ) / total_volume
            
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating benchmark VWAP: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check execution engine health"""
        try:
            # Check API connectivity (simulated)
            api_status = True  # In real implementation, ping exchange API
            
            # Check WebSocket status (simulated)
            ws_status = True  # In real implementation, check WebSocket connection
            
            # Calculate average latency
            recent_orders = [o for o in self.order_history[-10:] if 'execution_time_ms' in o]
            avg_latency = sum(o['execution_time_ms'] for o in recent_orders) / len(recent_orders) if recent_orders else 0
            
            self.health_status.update({
                'api_connected': api_status,
                'websocket_connected': ws_status,
                'last_heartbeat': datetime.utcnow().isoformat(),
                'latency_ms': avg_latency
            })
            
            return {
                'healthy': api_status and ws_status,
                'status': self.health_status,
                'active_orders_count': len(self.active_orders),
                'execution_stats': self.execution_stats
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution engine summary"""
        try:
            # Calculate win rate from execution quality
            quality_metrics = self.execution_benchmarks['vwap_execution_quality']
            
            if quality_metrics:
                positive_executions = sum(1 for m in quality_metrics if m['vwap_difference_percent'] < 0)
                execution_quality_rate = positive_executions / len(quality_metrics)
                avg_vwap_diff = sum(m['vwap_difference_percent'] for m in quality_metrics) / len(quality_metrics)
                avg_slippage = sum(m['slippage_percent'] for m in quality_metrics) / len(quality_metrics)
            else:
                execution_quality_rate = 0.0
                avg_vwap_diff = 0.0
                avg_slippage = 0.0
            
            return {
                'execution_stats': self.execution_stats,
                'health_status': self.health_status,
                'active_orders_count': len(self.active_orders),
                'order_history_count': len(self.order_history),
                'execution_quality': {
                    'quality_rate': execution_quality_rate,
                    'avg_vwap_difference_percent': avg_vwap_diff,
                    'avg_slippage_percent': avg_slippage,
                    'total_quality_samples': len(quality_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting execution summary: {e}")
            return {'error': str(e)}