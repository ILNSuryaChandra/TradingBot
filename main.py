#!/usr/bin/env python3
from trading_bot.setup import TradingBotSetup
import asyncio
import logging
import signal
import yaml
from pathlib import Path
import sys

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout for console output
    ]
)

# Force UTF-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

async def shutdown(trader, loop):
    """Handle graceful shutdown"""
    logger.info("Initiating shutdown sequence...")
    await trader.stop()
    
    # Cancel all tasks except the current one
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    logger.info("Waiting for tasks to complete...")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()
    logger.info("Shutdown complete")

def handle_exception(loop, context):
    """Global exception handler"""
    exception = context.get('exception', None)
    msg = context.get('message')
    
    if exception:
        # Get full exception details
        if isinstance(exception, (KeyboardInterrupt, SystemExit)):
            return  # Let these exceptions propagate normally
            
        error_msg = f"Caught exception: {exception.__class__.__name__}: {str(exception)}"
        if hasattr(exception, '__traceback__'):
            import traceback
            error_msg += f"\nTraceback:\n{''.join(traceback.format_tb(exception.__traceback__))}"
    else:
        error_msg = f"Caught exception: {msg}"
        
    logger.error(error_msg)
    asyncio.create_task(shutdown(loop))

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
        
        # Initial validation
        if not setup.validate_configuration():
            logger.error("Invalid configuration. Exiting...")
            return
            
        # Test API connection
        if not await setup.test_api_connection():
            logger.error("API connection test failed. Exiting...")
            return
            
        # Initialize components
        logger.info("Initializing components...")
        components = setup.initialize_components()
        
        # Get the trader instance
        trader = components['trader']
        
        # Set up event loop with proper exception handling
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(handle_exception)
        
        # Set up signal handlers for graceful shutdown
        if sys.platform != 'win32':
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(shutdown(trader, loop))
                )
        else:
            # Windows-specific handling
            signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(shutdown(trader, loop)))
            signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(shutdown(trader, loop)))
        
        # Start trading
        logger.info("Starting trading operations...")
        await trader.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt...")
        if 'trader' in locals() and 'loop' in locals():
            await shutdown(trader, loop)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        if 'trader' in locals() and 'loop' in locals():
            await shutdown(trader, loop)
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