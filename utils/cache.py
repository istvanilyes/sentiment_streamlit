"""
Caching utilities for sentiment analysis to avoid duplicate API calls.
"""
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict
import config


class SentimentCache:
    """
    Simple file-based cache for sentiment analysis results.
    Uses MD5 hash of text as key to handle long comments.
    """

    def __init__(self, cache_file: str = "sentiment_cache.json"):
        """
        Initialize cache.

        Args:
            cache_file: Name of cache file
        """
        self.cache_path = config.CACHE_DIR / cache_file
        self.cache: Dict[str, str] = {}
        self.load()

    def _hash_text(self, text: str) -> str:
        """
        Create MD5 hash of text for use as cache key.

        Args:
            text: Input text

        Returns:
            MD5 hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[str]:
        """
        Get sentiment from cache.

        Args:
            text: Input text

        Returns:
            Cached sentiment or None if not found
        """
        if not text or not isinstance(text, str):
            return None

        key = self._hash_text(text.strip().lower())
        return self.cache.get(key)

    def set(self, text: str, sentiment: str):
        """
        Store sentiment in cache.

        Args:
            text: Input text
            sentiment: Sentiment result
        """
        if not text or not isinstance(text, str):
            return

        key = self._hash_text(text.strip().lower())
        self.cache[key] = sentiment

    def load(self):
        """Load cache from file."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def save(self):
        """Save cache to file."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()

    def size(self) -> int:
        """
        Get number of cached items.

        Returns:
            Number of cached sentiment results
        """
        return len(self.cache)

    def __len__(self):
        return self.size()

    def __contains__(self, text: str) -> bool:
        """Check if text is in cache."""
        return self.get(text) is not None
