# trading_bot/setup.py
import asyncio
from pathlib import Path
import yaml
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv

from .core.base import AsyncBybitClient
from .core.risk_management import RiskManager
from .models.ml_models import ModelEnsemble, TradingStrategy
from .core.trader import AutonomousTrader

__version__ = '0.1.0'

class TradingBotSetup:
    def __init__(self, config_path: str):
        self.version = __version__
        # Set base path to the project root directory
        self.base_path = Path(__file__).parent.parent
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Load environment variables
        self._load_environment()
        
        # Load config after logger is set up
        self.config = self._load_config()
        
    def _load_environment(self):
        """Load environment variables from .env file"""
        try:
            env_path = self.base_path / '.env'
            if not env_path.exists():
                self.logger.warning(".env file not found at: %s", env_path)
            load_dotenv(env_path)
            
            # Check required environment variables
            required_env_vars = ['BYBIT_API_KEY', 'BYBIT_API_SECRET']
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {missing_vars}")
                
            self.logger.info("Environment variables loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading environment variables: {str(e)}")
            raise
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from yaml file"""
        try:
            config_file = self.base_path / self.config_path
            self.logger.info(f"Loading config from: {config_file}")
            
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found at: {config_file}")
            
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                
            # Inject environment variables
            config['api']['api_key'] = os.getenv('BYBIT_API_KEY')
            config['api']['api_secret'] = os.getenv('BYBIT_API_SECRET')
            
            # Validate required sections
            required_sections = ['api', 'trading', 'models', 'strategy', 'monitoring', 'risk_management']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                raise ValueError(f"Missing required config sections: {missing_sections}")
                
            self.logger.info("Config loaded and validated successfully")
            return config
            
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML config: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise
            
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.INFO)
        
        # Avoid adding duplicate handlers
        if not logger.handlers:
            # Create logs directory if it doesn't exist
            log_dir = self.base_path / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler
            fh = logging.FileHandler(log_dir / 'trading_bot.log')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # Add handlers
            logger.addHandler(fh)
            logger.addHandler(ch)
            
        return logger
        
    def setup_directory_structure(self):
        """Create necessary directories for the trading bot"""
        try:
            # Create main directories
            directories = [
                'logs',
                'data',
                'models',
                'state',
                'state/backups',
                'analytics'
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                
            self.logger.info("Directory structure created successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating directory structure: {str(e)}")
            return False
            
    def validate_configuration(self) -> bool:
        """Validate the configuration file and test API connectivity"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Check required sections
            required_sections = [
                'api',
                'trading',
                'models',
                'strategy',
                'monitoring',
                'risk_management'
            ]
            
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required config section: {section}")
                    return False
                    
            # Validate API configuration
            if not all(k in config['api'] for k in ['testnet', 'base_url', 'api_key', 'api_secret']):
                self.logger.error("Invalid API configuration")
                return False
                
            # Test API connectivity
            client = AsyncBybitClient(config)
            try:
                # Create event loop for testing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Test API connection
                response = loop.run_until_complete(client._make_request(
                    lambda: client.client.get_wallet_balance(
                        accountType="UNIFIED",
                        coin="USDT"
                    )
                ))
                
                if response.get('retCode') != 0:
                    self.logger.error(f"API connection test failed: {response.get('retMsg')}")
                    return False
                    
                self.logger.info("API connection test successful")
                
            except Exception as e:
                self.logger.error(f"API connection test failed: {str(e)}")
                return False
            finally:
                loop.close()
                
            # Validate trading configuration
            if not all(k in config['trading'] for k in ['symbols', 'timeframes', 'interval_seconds']):
                self.logger.error("Invalid trading configuration")
                return False
                
            self.logger.info("Config loaded and validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {str(e)}")
            return False
            
    def initialize_components(self) -> Dict[str, Any]:
        """Initialize all trading bot components"""
        try:
            self.logger.info("Starting component initialization...")
            
            # Load configuration
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Initialize client
            client = AsyncBybitClient(config)
            self.logger.info("Client initialized")
            
            # Initialize risk manager
            risk_manager = RiskManager(config, client)
            self.logger.info("Risk manager initialized")
            
            # Initialize model ensemble
            model_ensemble = ModelEnsemble(config)
            self.logger.info("Model ensemble initialized")
            
            # Initialize strategy
            strategy = TradingStrategy(config, model_ensemble)
            self.logger.info("Strategy initialized")
            
            # Initialize trader
            trader = AutonomousTrader(config)
            self.logger.info("Trader initialized")
            
            components = {
                'client': client,
                'risk_manager': risk_manager,
                'model_ensemble': model_ensemble,
                'strategy': strategy,
                'trader': trader
            }
            
            self.logger.info("All components initialized successfully")
            return components
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise