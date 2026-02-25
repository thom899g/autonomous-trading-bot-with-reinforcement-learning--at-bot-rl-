# Autonomous Trading Bot with Reinforcement Learning (AT-Bot RL)

## Objective
**TITLE:** Autonomous Trading Bot with Reinforcement Learning (AT-Bot RL)

**DESCRIPTION:**  
An AI-driven trading system utilizing reinforcement learning to adapt and optimize trading strategies

## Strategy
Research and implement using available tools.

## Execution Output
SUMMARY: I've architected and implemented core components of the Autonomous Trading Bot with Reinforcement Learning (AT-Bot RL), focusing on the foundational trading environment, data management, and RL infrastructure. The system follows strict architectural rigor with comprehensive error handling, type hints, logging, and Firebase integration for state management.

OUTPUT:

### FILE: requirements.txt
```txt
# Core dependencies for AT-Bot RL
ccxt==4.2.0
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.4.0
gymnasium==0.29.0
stable-baselines3==2.1.0
firebase-admin==6.2.0
python-dotenv==1.0.0
schedule==1.2.1
ta==0.11.0
matplotlib==3.8.0
```

### FILE: config.py
```python
"""
Configuration management for AT-Bot RL.
Centralizes all configuration with validation and type safety.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from dotenv import load_dotenv

load_dotenv()

class ExchangeType(Enum):
    """Supported cryptocurrency exchanges"""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"

class TradingMode(Enum):
    """Trading environment modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation"""
    project_id: str
    credentials_path: str = "firebase_credentials.json"
    collection_name: str = "trading_bot"
    
    def __post_init__(self):
        if not self.project_id:
            raise ValueError("Firebase project_id must be provided")
        if not os.path.exists(self.credentials_path):
            logging.warning(f"Firebase credentials not found at {self.credentials_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "credentials_path": self.credentials_path,
            "collection_name": self.collection_name
        }

@dataclass
class ExchangeConfig:
    """Exchange configuration with safety defaults"""
    exchange_type: ExchangeType
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True  # Default to sandbox for safety
    
    def __post_init__(self):
        # Validate API credentials only needed for live trading
        if not self.sandbox and (not self.api_key or not self.api_secret):
            logging.warning("Live trading requires both API key and secret")
    
    def get_ccxt_config(self) -> Dict[str, Any]:
        """Convert to CCXT-compatible config"""
        config = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
        if self.sandbox:
            config["options"]["test"] = True  # Use testnet/sandbox
        if self.api_key and self.api_secret:
            config["apiKey"] = self.api_key
            config["secret"] = self.api_secret
        return config

@dataclass
class TradingConfig:
    """Trading parameters with validation"""
    symbol: str  # e.g., "BTC/USDT"
    timeframe: str  # e.g., "1h"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of balance
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    
    def __post_init__(self):
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        if self.stop_loss_pct <= 0 or self.take_profit_pct <= 0:
            raise ValueError("Stop loss and take profit must be positive")

class ConfigManager:
    """Manages all configuration with singleton pattern"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize configuration from environment and defaults"""
        self.trading_mode = TradingMode(
            os.getenv("TRADING_MODE", "paper").upper()
        )
        
        self.firebase = FirebaseConfig(
            project_id=os.getenv("FIREBASE_PROJECT_ID", "at-bot-rl"),
            credentials_path=os.getenv("FIREBASE_CREDS_PATH", "firebase_credentials.json"),
            collection_name=os.getenv("FIREBASE_COLLECTION", "trading_bot")
        )
        
        self.exchange = ExchangeConfig(
            exchange_type=ExchangeType(
                os.getenv("EXCHANGE_TYPE", "binance").upper()
            ),
            api_key=os.getenv("EXCHANGE_API_KEY"),
            api_secret=os.getenv("EXCHANGE_API_SECRET"),
            sandbox=self.trading_mode != TradingMode.LIVE
        )
        
        self.trading = TradingConfig(
            symbol=os.getenv("TRADING_SYMBOL", "BTC/USDT"),
            timeframe=os.getenv("TRADING_TIMEFRAME", "1h"),
            initial_balance=float(os.getenv("INITIAL_BALANCE", "10000.0")),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.1")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.02")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
        )
        
        # RL Training configuration
        self.rl_training = {
            "total_timesteps": int(os.getenv("RL_TOTAL_TIMESTEPS", "100000")),
            "learning_rate": float(os.getenv("RL_LEARNING_RATE", "0.0003")),
            "gamma": float