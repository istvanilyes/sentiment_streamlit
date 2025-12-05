"""
Sentiment analysis service using OpenAI API with async concurrency.
"""
import asyncio
from typing import Optional, Dict
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import config


class SentimentAnalyzer:
    """
    Handles sentiment analysis using OpenAI API with concurrent processing.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize sentiment analyzer.

        Args:
            api_key: OpenAI API key (uses config if not provided)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        # In sentiment_analyzer.py, in __init__ method, after line 28:
        print(f"API Key loaded: {self.api_key[:10]}..." if self.api_key else "API Key is None!")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = config.OPENAI_MODEL

        # System prompt for sentiment classification
        self.system_prompt = """You are a sentiment analyzer. Classify the given text as one of the following:
- Positive
- Neutral
- Negative

Respond with ONLY ONE WORD: either "Positive", "Neutral", or "Negative"."""

    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=config.RETRY_MULTIPLIER,
            min=config.RETRY_DELAY,
            max=10
        ),
        retry=retry_if_exception_type((Exception,))
    )
    async def analyze_single(self, text: str) -> str:
        """
        Analyze sentiment of a single text with retry logic.

        Args:
            text: Text to analyze

        Returns:
            Sentiment classification (Positive, Neutral, or Negative)
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=10,
                temperature=0  # Deterministic responses
            )

            sentiment = response.choices[0].message.content.strip()

            # Validate response
            if sentiment in config.SENTIMENT_CATEGORIES:
                return sentiment
            else:
                # If response is not valid, try to map it
                sentiment_lower = sentiment.lower()
                if 'positive' in sentiment_lower:
                    return "Positive"
                elif 'negative' in sentiment_lower:
                    return "Negative"
                else:
                    return "Neutral"

        except Exception as e:
            # If all retries fail, return error marker
            print(f"Error analyzing text: {str(e)}")
            raise

    async def analyze_batch(
        self,
        texts: list[str],
        cache: Optional[Dict[str, str]] = None,
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> list[str]:
        """
        Analyze sentiment of multiple texts concurrently.

        Args:
            texts: List of texts to analyze
            cache: Optional cache dictionary for deduplication
            semaphore: Optional semaphore for rate limiting

        Returns:
            List of sentiment classifications
        """
        if semaphore is None:
            semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

        async def analyze_with_semaphore(text: str, index: int) -> tuple[int, str]:
            """Analyze with rate limiting."""
            # Check cache first
            if cache is not None and text in cache:
                return index, cache[text]

            async with semaphore:
                try:
                    sentiment = await self.analyze_single(text)

                    # Store in cache
                    if cache is not None:
                        cache[text] = sentiment

                    return index, sentiment

                except Exception as e:
                    # Return error marker if analysis fails
                    print(f"Failed to analyze text at index {index}: {str(e)}")
                    return index, "Error"

        # Create tasks for all texts
        tasks = [
            analyze_with_semaphore(text, i)
            for i, text in enumerate(texts)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Sort results by index and extract sentiments
        sorted_results = sorted(
            [r for r in results if not isinstance(r, Exception)],
            key=lambda x: x[0]
        )
        sentiments = [sentiment for _, sentiment in sorted_results]

        return sentiments

    async def analyze_batch_with_progress(
        self,
        texts: list[str],
        cache: Optional[Dict[str, str]] = None,
        progress_callback=None,
        semaphore: Optional[asyncio.Semaphore] = None
    ) -> list[str]:
        """
        Analyze sentiment of multiple texts concurrently with progress tracking.

        Args:
            texts: List of texts to analyze
            cache: Optional cache dictionary for deduplication
            progress_callback: Optional callback function(completed, total, sentiment)
            semaphore: Optional semaphore for rate limiting

        Returns:
            List of sentiment classifications
        """
        if semaphore is None:
            semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

        results = [""] * len(texts)
        completed = 0

        async def analyze_with_progress(text: str, index: int):
            """Analyze with progress tracking."""
            nonlocal completed

            # Check cache first
            if cache is not None and text in cache:
                sentiment = cache[text]
                results[index] = sentiment
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(texts), sentiment, True)
                return

            async with semaphore:
                try:
                    sentiment = await self.analyze_single(text)

                    # Store in cache
                    if cache is not None:
                        cache[text] = sentiment

                    results[index] = sentiment
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(texts), sentiment, False)

                except Exception as e:
                    print(f"Failed to analyze text at index {index}: {str(e)}")
                    results[index] = f"Error: {str(e)[:50]}"
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, len(texts), "Error", False)

        # Create tasks for all texts
        tasks = [
            analyze_with_progress(text, i)
            for i, text in enumerate(texts)
        ]

        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def close(self):
        """Close the API client."""
        # AsyncOpenAI client handles cleanup automatically
        pass
