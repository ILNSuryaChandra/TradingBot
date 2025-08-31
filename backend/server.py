from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto Trading Bot API",
    description="Production-grade crypto trading bot with VWAP strategies",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import services (will be created)
from services.data_service import DataService
from services.indicator_engine import IndicatorEngine
from services.strategy_engine import StrategyEngine
from services.risk_manager import RiskManager
from services.execution_engine import ExecutionEngine
from services.backtest_engine import BacktestEngine

# Initialize services
data_service = None
indicator_engine = None
strategy_engine = None
risk_manager = None
execution_engine = None
backtest_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global data_service, indicator_engine, strategy_engine, risk_manager, execution_engine, backtest_engine
    
    try:
        logger.info("Initializing trading bot services...")
        
        # Initialize core services
        data_service = DataService()
        indicator_engine = IndicatorEngine()
        risk_manager = RiskManager()
        strategy_engine = StrategyEngine(indicator_engine, risk_manager)
        execution_engine = ExecutionEngine(data_service, risk_manager)
        backtest_engine = BacktestEngine(data_service, indicator_engine, strategy_engine)
        
        logger.info("Trading bot services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active",
        "service": "Crypto Trading Bot API",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check database connection
        db_status = await data_service.health_check() if data_service else False
        
        return {
            "status": "healthy" if db_status else "degraded",
            "services": {
                "database": "connected" if db_status else "disconnected",
                "data_service": "active" if data_service else "inactive",
                "indicator_engine": "active" if indicator_engine else "inactive",
                "strategy_engine": "active" if strategy_engine else "inactive",
                "execution_engine": "active" if execution_engine else "inactive"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/api/pairs")
async def get_available_pairs():
    """Get available trading pairs"""
    return {
        "pairs": ["ETH/USDT", "SOL/USDT"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "exchanges": ["binance", "lighter", "hyperliquid"]
    }

@app.get("/api/historical/{pair}")
async def get_historical_data(pair: str, timeframe: str = "1h", limit: int = 100):
    """Get historical OHLCV data"""
    try:
        if not data_service:
            raise HTTPException(status_code=500, detail="Data service not initialized")
        
        data = await data_service.get_historical_ohlcv(
            pair=pair,
            timeframe=timeframe,
            limit=limit
        )
        
        return {
            "pair": pair,
            "timeframe": timeframe,
            "data": data,
            "count": len(data) if data else 0
        }
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicators/{pair}")
async def get_indicators(pair: str, timeframe: str = "1h", limit: int = 100):
    """Get technical indicators for a pair"""
    try:
        if not data_service or not indicator_engine:
            raise HTTPException(status_code=500, detail="Services not initialized")
        
        # Get historical data
        ohlcv_data = await data_service.get_historical_ohlcv(
            pair=pair,
            timeframe=timeframe,
            limit=limit
        )
        
        if not ohlcv_data:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Calculate indicators
        indicators = indicator_engine.calculate_all_indicators(ohlcv_data)
        
        return {
            "pair": pair,
            "timeframe": timeframe,
            "indicators": indicators,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def run_backtest(request: dict):
    """Run strategy backtest"""
    try:
        if not backtest_engine:
            raise HTTPException(status_code=500, detail="Backtest engine not initialized")
        
        pair = request.get("pair", "ETH/USDT")
        timeframe = request.get("timeframe", "1h")
        start_date = request.get("start_date")
        end_date = request.get("end_date")
        strategy_config = request.get("strategy_config", {})
        
        results = await backtest_engine.run_backtest(
            pair=pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_config=strategy_config
        )
        
        return {
            "backtest_results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)