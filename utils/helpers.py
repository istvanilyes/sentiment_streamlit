"""
Helper utility functions for the sentiment analysis application.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple
import config


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Uses a simple heuristic: ~1.3 tokens per word.

    Args:
        text: Input text string

    Returns:
        Estimated token count
    """
    if pd.isna(text) or not isinstance(text, str):
        return 0
    words = len(str(text).split())
    return int(words * 1.3)


def estimate_cost(total_input_tokens: int, total_output_tokens: int) -> float:
    """
    Estimate the cost of API calls based on token counts.

    Args:
        total_input_tokens: Total input tokens
        total_output_tokens: Total output tokens

    Returns:
        Estimated cost in USD
    """
    input_cost = (total_input_tokens / 1_000_000) * config.COST_INPUT_PER_1M
    output_cost = (total_output_tokens / 1_000_000) * config.COST_OUTPUT_PER_1M
    return input_cost + output_cost


def estimate_processing_time(num_rows: int, concurrent_requests: int = None) -> int:
    """
    Estimate processing time in seconds.

    Args:
        num_rows: Number of rows to process
        concurrent_requests: Number of concurrent requests (default from config)

    Returns:
        Estimated time in seconds
    """
    if concurrent_requests is None:
        concurrent_requests = config.MAX_CONCURRENT_REQUESTS

    # Assume ~1.2 seconds per request on average
    time_per_batch = 1.2
    num_batches = (num_rows + concurrent_requests - 1) // concurrent_requests
    return int(num_batches * time_per_batch)


def format_time(seconds: int) -> str:
    """
    Format seconds into a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds}s"

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def generate_output_filename(original_filename: str, suffix: str = "with_sentiment") -> str:
    """
    Generate output filename based on original filename.

    Args:
        original_filename: Original uploaded filename
        suffix: Suffix to add to filename

    Returns:
        New filename with timestamp and suffix
    """
    path = Path(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{path.stem}_{suffix}_{timestamp}{path.suffix}"


def validate_dataframe(df: pd.DataFrame, selected_column: str) -> Tuple[bool, str]:
    """
    Validate that dataframe and selected column are valid for processing.

    Args:
        df: Input dataframe
        selected_column: Selected text column name

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "Dataframe is empty"

    if selected_column not in df.columns:
        return False, f"Column '{selected_column}' not found in dataframe"

    # Check if column has any non-null text values
    valid_texts = df[selected_column].dropna()
    if len(valid_texts) == 0:
        return False, f"Column '{selected_column}' has no valid text values"

    return True, ""


def clean_text(text: str) -> str:
    """
    Clean and preprocess text before sentiment analysis.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Strip whitespace
    text = str(text).strip()

    # Remove excessive whitespace
    text = " ".join(text.split())

    return text


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)
