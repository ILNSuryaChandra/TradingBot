# trading_bot/setup.py
from pathlib import Path
import yaml
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv

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
        """Create necessary directories"""
        try:
            directories = ['data', 'logs', 'models', 'results', 'configs']
            
            for directory in directories:
                dir_path = self.base_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
            self.logger.info("Directory structure created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating directory structure: {str(e)}")
            raise
            
    def validate_configuration(self) -> bool:
        """Validate the configuration file"""
        try:
            # Validate API configuration
            api_config = self.config['api']
            required_api_fields = ['testnet', 'base_url', 'rate_limit_margin', 'api_key', 'api_secret']
            
            for field in required_api_fields:
                if not api_config.get(field):
                    self.logger.error(f"Missing or empty required API field: {field}")
                    return False
                    
            # Validate trading configuration
            trading_config = self.config['trading']
            required_trading_fields = ['interval_seconds', 'symbols', 'timeframes', 'position_management']
            
            for field in required_trading_fields:
                if field not in trading_config:
                    self.logger.error(f"Missing required trading field: {field}")
                    return False
                    
            # Validate model configuration
            models_config = self.config['models']
            required_model_fields = ['training']
            
            for field in required_model_fields:
                if field not in models_config:
                    self.logger.error(f"Missing required model field: {field}")
                    return False
                    
            self.logger.info("Configuration validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
            
    def initialize_components(self) -> Dict[str, Any]:
        """Initialize all trading bot components"""
        from .core.trader import AutonomousTrader
        from .core.risk_management import RiskManager
        from .core.base import AsyncBybitClient
        from .models.ml_models import ModelEnsemble
        from .custom.strategy_integration import AdvancedStrategyIntegration
        
        try:
            self.logger.info("Starting component initialization...")
            components = {}
            
            # Verify config is loaded
            if not self.config:
                raise ValueError("Config not loaded")
                
            # Initialize each component with detailed logging
            components['client'] = AsyncBybitClient(self.config)
            self.logger.info("Client initialized")
            
            components['risk_manager'] = RiskManager(self.config, components['client'])
            self.logger.info("Risk manager initialized")
            
            components['model_ensemble'] = ModelEnsemble(self.config)
            self.logger.info("Model ensemble initialized")
            
            components['strategy'] = AdvancedStrategyIntegration(self.config)
            self.logger.info("Strategy initialized")
            
            components['trader'] = AutonomousTrader(self.config)
            self.logger.info("Trader initialized")
            
            self.logger.info("All components initialized successfully")
            return components
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            if self.config:
                self.logger.error(f"Available config sections: {list(self.config.keys())}")
            raise