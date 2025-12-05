"""
Sentiment Analysis Application using OpenAI API
Main Streamlit interface for uploading files and analyzing sentiment.
"""
import streamlit as st
import pandas as pd
import asyncio
from pathlib import Path
from datetime import datetime
import config
from services.file_parser import FileParser
from services.processor import SentimentProcessor
from utils.helpers import (
    estimate_tokens,
    estimate_cost,
    estimate_processing_time,
    format_time,
    generate_output_filename,
    validate_dataframe
)


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'upload_filename' not in st.session_state:
        st.session_state.upload_filename = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None


def display_header():
    """Display application header."""
    st.title("📊 Sentiment Analysis Application")
    st.markdown("Upload your CSV or Excel file to analyze sentiment of comments using OpenAI GPT-4o-mini")
    st.markdown("---")


def display_file_upload():
    """Display file upload section."""
    st.subheader("1️⃣ Upload Your File")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing comments or text to analyze"
    )

    if uploaded_file is not None:
        # Save uploaded file
        upload_path = config.UPLOADS_DIR / uploaded_file.name
        with open(upload_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Parse file
        df, error = FileParser.read_file(str(upload_path))

        if error:
            st.error(f"Error reading file: {error}")
            return None, None

        st.session_state.df = df
        st.session_state.upload_filename = uploaded_file.name
        st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

        return df, uploaded_file.name

    return None, None


def display_file_info(df: pd.DataFrame):
    """Display information about uploaded file."""
    st.subheader("📋 File Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        text_columns = FileParser.get_text_columns(df)
        st.metric("Text Columns", len(text_columns))

    # Preview
    with st.expander("📄 Preview First 5 Rows"):
        st.dataframe(df.head(), use_container_width=True)


def display_column_selection(df: pd.DataFrame) -> str:
    """Display column selection and return selected column."""
    st.subheader("2️⃣ Select Text Column")

    text_columns = FileParser.get_text_columns(df)

    if not text_columns:
        st.error("No text columns found in the file")
        return None

    selected_column = st.selectbox(
        "Choose the column containing text to analyze",
        options=text_columns,
        help="Select the column that contains comments or text for sentiment analysis"
    )

    # Show sample values
    if selected_column:
        st.write("**Sample values from selected column:**")
        sample_values = df[selected_column].dropna().head(3).tolist()
        for i, val in enumerate(sample_values, 1):
            with st.container():
                st.text(f"{i}. {str(val)[:200]}{'...' if len(str(val)) > 200 else ''}")

        # Show statistics
        total_rows = len(df)
        valid_rows = df[selected_column].notna().sum()
        unique_rows = FileParser.get_unique_count(df, selected_column)
        duplicate_rows = FileParser.get_duplicate_count(df, selected_column)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valid Rows", f"{valid_rows:,}")
        with col2:
            st.metric("Empty Rows", f"{total_rows - valid_rows:,}")
        with col3:
            st.metric("Unique Values", f"{unique_rows:,}")
        with col4:
            st.metric("Duplicates", f"{duplicate_rows:,}")

        if duplicate_rows > 0:
            savings_pct = (duplicate_rows / valid_rows) * 100
            st.info(f"💡 {duplicate_rows:,} duplicate comments will be cached, saving ~{savings_pct:.1f}% of API calls!")

    return selected_column


def display_cost_estimate(df: pd.DataFrame, text_column: str):
    """Display cost and time estimates."""
    st.subheader("3️⃣ Estimates")

    valid_texts = df[text_column].dropna()
    num_rows = len(valid_texts)

    # Estimate tokens
    total_input_tokens = sum(estimate_tokens(str(text)) for text in valid_texts)
    # Output is just one word per comment
    total_output_tokens = num_rows * 2  # ~2 tokens per sentiment word

    # Estimate cost
    estimated_cost = estimate_cost(total_input_tokens, total_output_tokens)

    # Estimate time
    estimated_time = estimate_processing_time(num_rows, config.MAX_CONCURRENT_REQUESTS)

    # Display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Comments to Analyze", f"{num_rows:,}")
    with col2:
        st.metric("Estimated Cost", f"${estimated_cost:.3f}")
    with col3:
        st.metric("Estimated Time", format_time(estimated_time))

    st.info(f"💰 Using model: **{config.OPENAI_MODEL}** | Concurrent requests: **{config.MAX_CONCURRENT_REQUESTS}**")


def display_processing_section(df: pd.DataFrame, text_column: str, filename: str):
    """Display processing section with start button."""
    st.subheader("4️⃣ Start Analysis")

    # Check API key
    if not config.OPENAI_API_KEY:
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        st.code("OPENAI_API_KEY=your_api_key_here")
        return

    # Validate
    is_valid, error_msg = validate_dataframe(df, text_column)
    if not is_valid:
        st.error(f"Validation error: {error_msg}")
        return

    # Start button
    if st.button("🚀 Start Sentiment Analysis", type="primary", use_container_width=True):
        st.session_state.processing = True
        process_sentiment_analysis(df, text_column, filename)


def process_sentiment_analysis(df: pd.DataFrame, text_column: str, filename: str):
    """Process sentiment analysis with progress tracking."""
    # Progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_container = st.empty()

    # Initialize processor
    processor = SentimentProcessor()

    # Progress callback
    def update_progress(current, total, stats):
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)

        status_text.text(f"Processing: {current:,} / {total:,} comments ({progress*100:.1f}%)")

        # Display stats
        with stats_container.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("✅ Processed", f"{stats['processed']:,}")
            with col2:
                st.metric("💚 Positive", f"{stats['positive']:,}")
            with col3:
                st.metric("😐 Neutral", f"{stats['neutral']:,}")
            with col4:
                st.metric("❌ Negative", f"{stats['negative']:,}")

            if stats['cached'] > 0:
                st.caption(f"🔄 Cached: {stats['cached']:,} | ⚠️ Errors: {stats['errors']:,}")

    # Process
    try:
        with st.spinner("Initializing sentiment analysis..."):
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result_df = loop.run_until_complete(
                processor.process_dataframe(
                    df=df,
                    text_column=text_column,
                    progress_callback=update_progress,
                    use_cache=True
                )
            )

            loop.close()

        st.session_state.processed_df = result_df
        st.session_state.processing = False

        # Success message
        st.success("✅ Sentiment analysis completed successfully!")

        # Final statistics
        st.subheader("📊 Results Summary")
        sentiment_counts = result_df['Sentiment'].value_counts()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total = len(result_df)
            st.metric("Total Rows", f"{total:,}")
        with col2:
            positive = sentiment_counts.get('Positive', 0)
            positive_pct = (positive / total * 100) if total > 0 else 0
            st.metric("Positive", f"{positive:,}", f"{positive_pct:.1f}%")
        with col3:
            neutral = sentiment_counts.get('Neutral', 0)
            neutral_pct = (neutral / total * 100) if total > 0 else 0
            st.metric("Neutral", f"{neutral:,}", f"{neutral_pct:.1f}%")
        with col4:
            negative = sentiment_counts.get('Negative', 0)
            negative_pct = (negative / total * 100) if total > 0 else 0
            st.metric("Negative", f"{negative:,}", f"{negative_pct:.1f}%")

        # Show preview
        with st.expander("👀 Preview Results (First 10 Rows)"):
            preview_cols = [text_column, 'Sentiment']
            st.dataframe(result_df[preview_cols].head(10), use_container_width=True)

        # Download button
        display_download_section(result_df, filename)

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.session_state.processing = False

    finally:
        processor.close()


def display_download_section(df: pd.DataFrame, original_filename: str):
    """Display download section."""
    st.subheader("5️⃣ Download Results")

    # Generate output filename
    output_filename = generate_output_filename(original_filename)
    output_path = config.RESULTS_DIR / output_filename

    # Save file
    extension = Path(original_filename).suffix.lower()
    success, error = FileParser.save_dataframe(df, str(output_path))

    if not success:
        st.error(f"Error saving file: {error}")
        return

    # Read file for download
    with open(output_path, 'rb') as f:
        file_data = f.read()

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download Results (CSV/Excel)",
            data=file_data,
            file_name=output_filename,
            mime='application/octet-stream',
            use_container_width=True
        )

    with col2:
        # Also offer CSV download
        csv_data = df.to_csv(index=False).encode('utf-8')
        csv_filename = output_filename.replace('.xlsx', '.csv').replace('.xls', '.csv')

        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=csv_filename,
            mime='text/csv',
            use_container_width=True
        )


def main():
    """Main application flow."""
    init_session_state()
    display_header()

    # File upload
    df, filename = display_file_upload()

    # If file uploaded, show next steps
    if df is not None:
        display_file_info(df)

        # Column selection
        selected_column = display_column_selection(df)

        if selected_column:
            # Cost estimate
            display_cost_estimate(df, selected_column)

            # Processing section
            display_processing_section(df, selected_column, filename)

    # If already processed, show download
    elif st.session_state.processed_df is not None and not st.session_state.processing:
        st.info("Previous results are available for download")
        if st.button("🔄 Process New File"):
            st.session_state.processed_df = None
            st.session_state.df = None
            st.rerun()
        else:
            display_download_section(
                st.session_state.processed_df,
                st.session_state.upload_filename
            )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.write(f"**Model:** {config.OPENAI_MODEL}")
        st.write(f"**Max Concurrent:** {config.MAX_CONCURRENT_REQUESTS}")
        st.write(f"**Chunk Size:** {config.CHUNK_SIZE}")

        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This application uses OpenAI's GPT-4o-mini model to classify sentiment of text comments as:
        - 💚 **Positive**
        - 😐 **Neutral**
        - ❌ **Negative**

        **Features:**
        - Upload CSV or Excel files
        - Concurrent processing for speed
        - Caching to avoid duplicate API calls
        - Real-time progress tracking
        - Cost and time estimates
        """)

        st.markdown("---")
        st.caption("Powered by OpenAI GPT-4o-mini")


if __name__ == "__main__":
    main()
