"""
Main processing logic for sentiment analysis with progress tracking.
"""
import asyncio
import pandas as pd
from typing import Optional, Callable, Dict, Tuple
from pathlib import Path
import config
from services.sentiment_analyzer import SentimentAnalyzer
from utils.cache import SentimentCache
from utils.helpers import clean_text


class SentimentProcessor:
    """
    Handles the main processing workflow for sentiment analysis.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize processor.

        Args:
            api_key: OpenAI API key (uses config if not provided)
        """
        self.analyzer = SentimentAnalyzer(api_key)
        self.cache = SentimentCache()

    def prepare_texts(self, df: pd.DataFrame, text_column: str) -> Tuple[list[str], list[int]]:
        """
        Prepare texts for processing, handling null values.

        Args:
            df: Input DataFrame
            text_column: Name of text column

        Returns:
            Tuple of (cleaned_texts, valid_indices)
        """
        texts = []
        valid_indices = []

        for idx, row in df.iterrows():
            text = row[text_column]

            # Skip null or empty values
            if pd.isna(text) or str(text).strip() == "":
                continue

            cleaned = clean_text(text)
            if cleaned:
                texts.append(cleaned)
                valid_indices.append(idx)

        return texts, valid_indices

    async def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        progress_callback: Optional[Callable] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Process entire DataFrame with sentiment analysis.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text to analyze
            progress_callback: Optional callback function(current, total, stats)
            use_cache: Whether to use caching for duplicate comments

        Returns:
            DataFrame with added 'Sentiment' column
        """
        # Create copy to avoid modifying original
        result_df = df.copy()

        # Initialize sentiment column
        result_df['Sentiment'] = 'N/A'

        # Prepare texts
        texts, valid_indices = self.prepare_texts(df, text_column)

        if not texts:
            return result_df

        # Statistics
        stats = {
            'total': len(texts),
            'processed': 0,
            'cached': 0,
            'errors': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0
        }

        # Progress callback wrapper
        def track_progress(completed, total, sentiment, from_cache):
            stats['processed'] = completed
            if from_cache:
                stats['cached'] += 1

            if sentiment == "Positive":
                stats['positive'] += 1
            elif sentiment == "Neutral":
                stats['neutral'] += 1
            elif sentiment == "Negative":
                stats['negative'] += 1
            elif sentiment == "Error":
                stats['errors'] += 1

            if progress_callback:
                progress_callback(completed, total, stats)

        # Get cache dict for batch processing
        cache_dict = {}
        if use_cache:
            # Pre-populate cache dict with existing cache entries
            for text in texts:
                cached = self.cache.get(text)
                if cached:
                    cache_dict[text] = cached

        # Process texts
        sentiments = await self.analyzer.analyze_batch_with_progress(
            texts=texts,
            cache=cache_dict if use_cache else None,
            progress_callback=track_progress
        )

        # Update DataFrame with results
        for idx, sentiment in zip(valid_indices, sentiments):
            result_df.at[idx, 'Sentiment'] = sentiment

        # Save cache
        if use_cache:
            # Update cache with new entries from cache_dict
            for text, sentiment in cache_dict.items():
                self.cache.set(text, sentiment)
            self.cache.save()

        return result_df

    async def process_dataframe_chunked(
        self,
        df: pd.DataFrame,
        text_column: str,
        chunk_size: int = None,
        progress_callback: Optional[Callable] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Process DataFrame in chunks for memory efficiency with large files.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text to analyze
            chunk_size: Size of each chunk (default from config)
            progress_callback: Optional callback function(current, total, stats)
            use_cache: Whether to use caching for duplicate comments

        Returns:
            DataFrame with added 'Sentiment' column
        """
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE

        # Create copy
        result_df = df.copy()
        result_df['Sentiment'] = 'N/A'

        # Prepare all texts
        texts, valid_indices = self.prepare_texts(df, text_column)

        if not texts:
            return result_df

        total = len(texts)

        # Overall statistics
        overall_stats = {
            'total': total,
            'processed': 0,
            'cached': 0,
            'errors': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0
        }

        # Get cache dict
        cache_dict = {}
        if use_cache:
            for text in texts:
                cached = self.cache.get(text)
                if cached:
                    cache_dict[text] = cached

        # Process in chunks
        for chunk_start in range(0, len(texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            chunk_indices = valid_indices[chunk_start:chunk_end]

            # Progress callback for this chunk
            def chunk_progress(completed, chunk_total, sentiment, from_cache):
                overall_stats['processed'] = chunk_start + completed

                if from_cache:
                    overall_stats['cached'] += 1

                if sentiment == "Positive":
                    overall_stats['positive'] += 1
                elif sentiment == "Neutral":
                    overall_stats['neutral'] += 1
                elif sentiment == "Negative":
                    overall_stats['negative'] += 1
                elif sentiment == "Error":
                    overall_stats['errors'] += 1

                if progress_callback:
                    progress_callback(overall_stats['processed'], total, overall_stats)

            # Process chunk
            chunk_sentiments = await self.analyzer.analyze_batch_with_progress(
                texts=chunk_texts,
                cache=cache_dict if use_cache else None,
                progress_callback=chunk_progress
            )

            # Update DataFrame
            for idx, sentiment in zip(chunk_indices, chunk_sentiments):
                result_df.at[idx, 'Sentiment'] = sentiment

        # Save cache
        if use_cache:
            for text, sentiment in cache_dict.items():
                self.cache.set(text, sentiment)
            self.cache.save()

        return result_df

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            'size': self.cache.size(),
            'path': str(self.cache.cache_path)
        }

    def clear_cache(self):
        """Clear the sentiment cache."""
        self.cache.clear()

    def close(self):
        """Clean up resources."""
        self.analyzer.close()
