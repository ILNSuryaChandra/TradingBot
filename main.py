from trading_bot.setup import TradingBotSetup
import asyncio
import logging
import signal
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        logger.info("Initializing Trading Bot...")
        
        # Load config directly first to validate
        config_path = Path('config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Config loaded with sections: {list(config.keys())}")
        
        # Initialize setup with both path and loaded config
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
        
        # Set up graceful shutdown
        def shutdown_handler():
            logger.info("Shutdown signal received...")
            asyncio.create_task(trader.stop())

        # Register shutdown handler
        for sig in [signal.SIGINT, signal.SIGTERM]:
            asyncio.get_event_loop().add_signal_handler(
                sig, 
                shutdown_handler
            )
        
        # Start trading
        logger.info("Starting trading...")
        await trader.start()
        
    except Exception as e:
        logger.error(f"Error in main: {repr(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot stopped due to error: {repr(e)}")