import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config():
    try:
        config_path = Path('config.yaml')
        logger.info(f"Looking for config at: {config_path.absolute()}")
        
        if not config_path.exists():
            logger.error(f"Config file not found at: {config_path.absolute()}")
            return
            
        logger.info("Config file found, attempting to load...")
        
        with open(config_path, 'r') as file:
            content = file.read()
            logger.info(f"Raw content length: {len(content)}")
            logger.info("First few characters:")
            logger.info(content[:100])
            
            # Try to parse
            config = yaml.safe_load(content)
            logger.info("Config loaded successfully")
            logger.info(f"Config sections: {list(config.keys())}")
            
            # Validate required sections
            required_sections = ['api', 'trading', 'models', 'strategy', 'monitoring', 'risk_management']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                logger.error(f"Missing required sections: {missing_sections}")
            else:
                logger.info("All required sections present")
                
    except Exception as e:
        logger.error(f"Error testing config: {str(e)}")

if __name__ == "__main__":
    test_config()