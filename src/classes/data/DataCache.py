"""
Data Cache Module

Provides a unified caching interface supporting both pickle files
and SQLite database storage with freshness checks.
"""

import os
import pickle
import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from classes.DatabaseManager import (
    connect_db,
    create_tables,
    get_last_date,
    insert_price_rows,
    insert_company_info,
    get_price_dataframe,
    get_company_info,
)


logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified caching interface for stock data.
    
    Supports:
    - Pickle files for daily snapshot caching
    - SQLite database for incremental updates
    - Configurable TTL (time-to-live) for cache freshness
    """
    
    DEFAULT_TTL_HOURS = 24
    
    def __init__(
        self,
        cache_dir: str = "data/historic_data",
        db_path: Optional[str] = None,
        ttl_hours: int = DEFAULT_TTL_HOURS,
    ):
        """
        Initialize the CacheManager.
        
        Args:
            cache_dir: Directory for pickle cache files
            db_path: Path to SQLite database (optional)
            ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = cache_dir
        self.db_path = db_path
        self.ttl = timedelta(hours=ttl_hours)
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str, market: str = "") -> str:
        """Generate cache file path with today's date."""
        timestamp = datetime.now().strftime("%y%m%d")
        subdir = os.path.join(self.cache_dir, market) if market else self.cache_dir
        Path(subdir).mkdir(parents=True, exist_ok=True)
        return os.path.join(subdir, f"{key}_{timestamp}.pkl")
    
    def _find_latest_cache(self, key: str, market: str = "") -> Optional[str]:
        """Find the most recent cache file for a given key."""
        subdir = os.path.join(self.cache_dir, market) if market else self.cache_dir
        
        if not os.path.exists(subdir):
            return None
        
        # Find files matching the key pattern
        matching = []
        for f in os.listdir(subdir):
            if f.startswith(f"{key}_") and f.endswith(".pkl"):
                matching.append(os.path.join(subdir, f))
        
        if not matching:
            return None
        
        # Return most recently modified
        return max(matching, key=os.path.getmtime)
    
    def is_fresh(self, key: str, market: str = "") -> bool:
        """
        Check if cache is fresh (within TTL).
        
        Args:
            key: Cache key
            market: Market subdirectory
            
        Returns:
            True if cache exists and is within TTL
        """
        cache_path = self._get_cache_path(key, market)
        
        # Check if today's cache exists
        if os.path.exists(cache_path):
            return True
        
        # Check for recent cache within TTL
        latest = self._find_latest_cache(key, market)
        if latest:
            mtime = datetime.fromtimestamp(os.path.getmtime(latest))
            if datetime.now() - mtime < self.ttl:
                return True
        
        return False
    
    def get(self, key: str, market: str = "") -> Optional[Dict]:
        """
        Load data from cache.
        
        Args:
            key: Cache key
            market: Market subdirectory
            
        Returns:
            Cached data or None if not found/stale
        """
        # First try today's cache
        cache_path = self._get_cache_path(key, market)
        
        if os.path.exists(cache_path):
            return self._load_pickle(cache_path)
        
        # Try latest cache within TTL
        latest = self._find_latest_cache(key, market)
        if latest:
            mtime = datetime.fromtimestamp(os.path.getmtime(latest))
            if datetime.now() - mtime < self.ttl:
                logger.info(f"Using recent cache from {latest}")
                return self._load_pickle(latest)
        
        return None
    
    def _load_pickle(self, path: str) -> Optional[Dict]:
        """Load data from pickle file."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded cache from {path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache from {path}: {e}")
            return None
    
    def set(self, key: str, data: Dict, market: str = "") -> bool:
        """
        Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
            market: Market subdirectory
            
        Returns:
            True if successful
        """
        cache_path = self._get_cache_path(key, market)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved cache to {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path}: {e}")
            return False
    
    def invalidate(self, key: str, market: str = "") -> int:
        """
        Remove all cache files for a key.
        
        Args:
            key: Cache key
            market: Market subdirectory
            
        Returns:
            Number of files removed
        """
        subdir = os.path.join(self.cache_dir, market) if market else self.cache_dir
        removed = 0
        
        if not os.path.exists(subdir):
            return 0
        
        for f in os.listdir(subdir):
            if f.startswith(f"{key}_") and f.endswith(".pkl"):
                try:
                    os.remove(os.path.join(subdir, f))
                    removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {f}: {e}")
        
        logger.info(f"Invalidated {removed} cache files for {key}")
        return removed
    
    # SQLite-specific methods
    
    def get_from_db(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load price data from SQLite database.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame or None
        """
        if not self.db_path:
            return None
        
        try:
            conn = connect_db(self.db_path)
            df = get_price_dataframe(conn, symbol)
            conn.close()
            return df if not df.empty else None
        except Exception as e:
            logger.warning(f"Failed to load {symbol} from database: {e}")
            return None
    
    def save_to_db(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        company_info: Optional[Dict] = None,
    ) -> bool:
        """
        Save stock data to SQLite database.
        
        Args:
            symbol: Stock symbol
            price_data: Price DataFrame
            company_info: Company information dict
            
        Returns:
            True if successful
        """
        if not self.db_path:
            return False
        
        try:
            conn = connect_db(self.db_path)
            create_tables(conn)
            insert_price_rows(conn, symbol, price_data)
            if company_info:
                insert_company_info(conn, symbol, company_info)
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Failed to save {symbol} to database: {e}")
            return False
    
    def get_db_last_date(self, symbol: str) -> Optional[str]:
        """
        Get the last date stored in the database for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Date string or None
        """
        if not self.db_path:
            return None
        
        try:
            conn = connect_db(self.db_path)
            last = get_last_date(conn, symbol)
            conn.close()
            return last
        except Exception as e:
            logger.warning(f"Failed to get last date for {symbol}: {e}")
            return None
    
    def get_company_info_from_db(self, symbol: str) -> Optional[Dict]:
        """
        Load company info from SQLite database.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company info dict or None
        """
        if not self.db_path:
            return None
        
        try:
            conn = connect_db(self.db_path)
            info = get_company_info(conn, symbol)
            conn.close()
            return info
        except Exception as e:
            logger.warning(f"Failed to load company info for {symbol}: {e}")
            return None
