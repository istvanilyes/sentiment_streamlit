# Quick Start Guide

## Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## First Test

A sample dataset is already included: `employee_review_mturk_dataset_test_v6_kaggle.csv`

**Steps:**
1. Click "Browse files" and select the CSV
2. Select "feedback" as the text column
3. Review the cost estimate (~$0.03 for full file)
4. Click "Start Sentiment Analysis"
5. Watch the progress bar
6. Download results when complete

## Using Your Own Data

**Requirements:**
- File format: CSV, XLSX, or XLS
- Must have at least one text column with comments
- Maximum recommended: 100K rows

**Example CSV format:**
```csv
id,comment,date
1,Great product!,2024-01-15
2,Not satisfied,2024-01-16
3,It's okay,2024-01-17
```

## Configuration

Edit `config.py` to adjust:

```python
MAX_CONCURRENT_REQUESTS = 50  # API concurrency
CHUNK_SIZE = 500              # Processing chunk size
OPENAI_MODEL = "gpt-4o-mini"  # Model to use
```

## Troubleshooting

**"OpenAI API key not found"**
- Make sure you created `.env` file with your API key
- Restart the Streamlit app after creating `.env`

**"Rate limit exceeded"**
- Reduce `MAX_CONCURRENT_REQUESTS` in config.py
- Wait a minute and try again
- Check your OpenAI API tier limits

**Slow processing**
- Check your internet connection
- Verify OpenAI API status: https://status.openai.com
- Reduce concurrent requests if on lower API tier

## Cost Management

**Tips to reduce costs:**
- The app automatically caches duplicate comments
- Use the cost estimator before processing
- Start with a small sample to test
- GPT-4o-mini is already the most cost-effective model

**Current pricing (GPT-4o-mini):**
- ~$0.01 per 1,000 comments (50 words each)
- ~$0.12 per 10,000 comments
- Actual cost often 30-50% lower due to caching

## Next Steps

- Review the full [README.md](README.md) for detailed documentation
- Customize sentiment categories if needed
- Adjust configuration for your use case
- Consider batch API for very large files (>100K rows)
