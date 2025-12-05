"""
File parsing service for CSV and Excel files.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import config


class FileParser:
    """
    Handles parsing of CSV and Excel files for sentiment analysis.
    """

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Get file extension.

        Args:
            filename: Name of file

        Returns:
            File extension (e.g., '.csv')
        """
        return Path(filename).suffix.lower()

    @staticmethod
    def is_supported_file(filename: str) -> bool:
        """
        Check if file type is supported.

        Args:
            filename: Name of file

        Returns:
            True if supported, False otherwise
        """
        extension = FileParser.get_file_extension(filename)
        return extension in config.SUPPORTED_EXTENSIONS

    @staticmethod
    def read_file(file_path: str, nrows: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Read CSV or Excel file into DataFrame.

        Args:
            file_path: Path to file
            nrows: Optional number of rows to read (for preview)

        Returns:
            Tuple of (DataFrame, error_message)
        """
        try:
            extension = FileParser.get_file_extension(file_path)

            if extension == '.csv':
                df = pd.read_csv(file_path, nrows=nrows, encoding='utf-8')
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=nrows)
            else:
                return None, f"Unsupported file type: {extension}"

            if df.empty:
                return None, "File is empty"

            return df, ""

        except UnicodeDecodeError:
            # Try different encodings for CSV
            try:
                df = pd.read_csv(file_path, nrows=nrows, encoding='latin-1')
                return df, ""
            except Exception as e:
                return None, f"Encoding error: {str(e)}"

        except Exception as e:
            return None, f"Error reading file: {str(e)}"

    @staticmethod
    def get_text_columns(df: pd.DataFrame) -> List[str]:
        """
        Get list of columns that contain text data (likely comment columns).

        Args:
            df: Input DataFrame

        Returns:
            List of column names with text data
        """
        text_columns = []

        for col in df.columns:
            # Check if column contains string data
            if df[col].dtype == 'object':
                # Sample first non-null value to verify it's text
                sample = df[col].dropna().head(1)
                if len(sample) > 0 and isinstance(sample.iloc[0], str):
                    text_columns.append(col)

        return text_columns

    @staticmethod
    def get_column_info(df: pd.DataFrame) -> dict:
        """
        Get information about DataFrame columns.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with column information
        """
        info = {
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'columns': []
        }

        for col in df.columns:
            non_null = df[col].notna().sum()
            col_info = {
                'name': col,
                'type': str(df[col].dtype),
                'non_null': non_null,
                'null': len(df) - non_null,
                'is_text': df[col].dtype == 'object'
            }
            info['columns'].append(col_info)

        return info

    @staticmethod
    def save_dataframe(df: pd.DataFrame, output_path: str) -> Tuple[bool, str]:
        """
        Save DataFrame to CSV or Excel file.

        Args:
            df: DataFrame to save
            output_path: Path to save file

        Returns:
            Tuple of (success, error_message)
        """
        try:
            extension = FileParser.get_file_extension(output_path)

            if extension == '.csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif extension in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False, engine='openpyxl')
            else:
                return False, f"Unsupported output file type: {extension}"

            return True, ""

        except Exception as e:
            return False, f"Error saving file: {str(e)}"

    @staticmethod
    def preview_dataframe(df: pd.DataFrame, num_rows: int = 5) -> pd.DataFrame:
        """
        Get preview of DataFrame.

        Args:
            df: Input DataFrame
            num_rows: Number of rows to preview

        Returns:
            Preview DataFrame
        """
        return df.head(num_rows)

    @staticmethod
    def get_unique_count(df: pd.DataFrame, column: str) -> int:
        """
        Get count of unique non-null values in column.

        Args:
            df: Input DataFrame
            column: Column name

        Returns:
            Count of unique values
        """
        return df[column].dropna().nunique()

    @staticmethod
    def get_duplicate_count(df: pd.DataFrame, column: str) -> int:
        """
        Get count of duplicate values in column.

        Args:
            df: Input DataFrame
            column: Column name

        Returns:
            Count of duplicate values (total duplicates, not unique duplicate values)
        """
        total = df[column].notna().sum()
        unique = FileParser.get_unique_count(df, column)
        return total - unique
