"""
Screener Registry

Provides discovery and management of available screeners.
"""

from typing import Dict, List, Optional, Type

from .base import BaseScreener


class ScreenerRegistry:
    """
    Registry for screener discovery and management.
    
    Usage:
        registry = ScreenerRegistry()
        registry.register(MACDScreener())
        
        screener = registry.get("macd")
        result = screener.screen(ticker, data)
    """
    
    _instance: Optional["ScreenerRegistry"] = None
    
    def __new__(cls) -> "ScreenerRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._screeners = {}
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "_screeners"):
            self._screeners: Dict[str, BaseScreener] = {}
    
    def register(self, screener: BaseScreener) -> None:
        """
        Register a screener.
        
        Args:
            screener: Screener instance to register
        """
        self._screeners[screener.name] = screener
    
    def unregister(self, name: str) -> None:
        """
        Unregister a screener.
        
        Args:
            name: Screener name to remove
        """
        if name in self._screeners:
            del self._screeners[name]
    
    def get(self, name: str) -> Optional[BaseScreener]:
        """
        Get a screener by name.
        
        Args:
            name: Screener name
            
        Returns:
            Screener instance or None if not found
        """
        return self._screeners.get(name)
    
    def list_available(self) -> List[str]:
        """Get list of registered screener names."""
        return list(self._screeners.keys())
    
    def list_all(self) -> List[BaseScreener]:
        """Get all registered screeners."""
        return list(self._screeners.values())
    
    def clear(self) -> None:
        """Clear all registered screeners."""
        self._screeners.clear()
    
    def __contains__(self, name: str) -> bool:
        return name in self._screeners
    
    def __len__(self) -> int:
        return len(self._screeners)
    
    def __repr__(self) -> str:
        return f"ScreenerRegistry({list(self._screeners.keys())})"


# Global registry instance
_registry = ScreenerRegistry()


def get_registry() -> ScreenerRegistry:
    """Get the global screener registry."""
    return _registry


def register_screener(screener: BaseScreener) -> BaseScreener:
    """
    Register a screener to the global registry.
    
    Can be used as a decorator:
        @register_screener
        class MyScreener(BaseScreener):
            ...
    """
    _registry.register(screener)
    return screener
