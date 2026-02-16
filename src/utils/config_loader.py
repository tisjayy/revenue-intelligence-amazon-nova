"""
Configuration loader for Revenue Intelligence Platform
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager with singleton pattern"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Default to config/config.yaml
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self._config
    
    @property
    def nyc_bounds(self):
        """Get NYC geographic boundaries"""
        return self.get('geographic.nyc_bounds')
    
    @property
    def n_clusters(self):
        """Get number of clusters"""
        return self.get('clustering.n_clusters', 40)
    
    @property
    def aws_region(self):
        """Get AWS region"""
        return self.get('aws.region', 'us-east-1')
    
    @property
    def nova_model_id(self):
        """Get Nova Lite model ID"""
        return self.get('aws.bedrock.nova_lite_model')


# Global config instance
config = Config()
