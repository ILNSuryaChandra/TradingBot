#!/usr/bin/env python3
from trading_bot.setup import TradingBotSetup
import asyncio
import logging
import signal
import yaml
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def shutdown(trader):
    """Handle graceful shutdown"""
    logger.info("Initiating shutdown sequence...")
    await trader.stop()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    logger.info("Waiting for tasks to complete...")
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Shutdown complete")

async def main():
    try:
        logger.info("Initializing Trading Bot...")
        
        # Load config
        config_path = Path('config.yaml')
        if not config_path.exists():
            logger.error("Configuration file not found")
            return
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Config loaded with sections: {list(config.keys())}")
        
        # Initialize setup
        setup = TradingBotSetup(str(config_path))
        
        # Create directory structure
        setup.setup_directory_structure()
        
        # Validate configuration
        if not setup.validate_configuration():
            logger.error("Invalid configuration. Exiting...")
            return
            
        # Initialize components
        logger.info("Initializing components...")
        components = setup.initialize_components()
        
        # Get the trader instance
        trader = components['trader']
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        signals = (signal.SIGTERM, signal.SIGINT)
        for sig in signals:
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(shutdown(trader))
            )
        
        # Start trading
        logger.info("Starting trading operations...")
        await trader.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt...")
        if 'trader' in locals():
            await shutdown(trader)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        if 'trader' in locals():
            await shutdown(trader)
        raise
    finally:
        logger.info("Trading bot shutdown complete")

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            # Set up proper event loop policy for Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {str(e)}")
        sys.exit(1)