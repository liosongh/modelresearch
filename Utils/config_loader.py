# =============================================================================
# Config Loader - 配置加载模块
# Version: v2.0
# =============================================================================

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    递归合并配置，override_config 覆盖 base_config。
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        merged: 合并后的配置
    """
    merged = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
            
    return merged


def get_model_config(config: Dict) -> Dict:
    """
    从完整配置中提取模型相关配置。
    """
    return {
        'lob_config': config.get('lob_encoder', {}),
        'trade_config': config.get('trade_encoder'),
        'fusion_config': config.get('fusion', {}),
        'transformer_config': config.get('transformer', {}),
        'output_config': config.get('output_head', {})
    }


def get_training_config(config: Dict) -> Dict:
    """
    从完整配置中提取训练相关配置。
    """
    return {
        'data': config.get('data', {}),
        'dataloader': config.get('dataloader', {}),
        'optimizer': config.get('optimizer', {}),
        'scheduler': config.get('scheduler', {}),
        'loss': config.get('loss', {}),
        'training': config.get('training', {}),
        'logging': config.get('logging', {}),
        'device': config.get('device', {})
    }


class ConfigManager:
    """
    配置管理器，支持多配置文件合并和验证。
    """
    
    def __init__(self, base_config_path: Optional[str] = None):
        self.config = {}
        
        if base_config_path:
            self.config = load_config(base_config_path)
            
    def load(self, config_path: str) -> 'ConfigManager':
        """加载配置文件。"""
        self.config = load_config(config_path)
        return self
        
    def merge(self, override_config: Dict) -> 'ConfigManager':
        """合并额外配置。"""
        self.config = merge_configs(self.config, override_config)
        return self
        
    def merge_from_file(self, config_path: str) -> 'ConfigManager':
        """从文件合并配置。"""
        override = load_config(config_path)
        return self.merge(override)
        
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> 'ConfigManager':
        """设置配置值。"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        return self
        
    @property
    def model_config(self) -> Dict:
        return get_model_config(self.config)
    
    @property
    def training_config(self) -> Dict:
        return get_training_config(self.config)
