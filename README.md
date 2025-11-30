# Construction Sentiment NLP

🏗️ **AI-Powered Sentiment Analysis for Construction Site Job Notes**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Overview

This application uses Natural Language Processing (NLP) to analyze construction site job notes and classify them as Positive, Neutral, or Negative. It helps project managers identify potential issues early and track project health through interactive dashboards.

## Features

### 📝 Sentiment Analyzer
- Real-time sentiment prediction for individual notes
- Color-coded results for quick understanding
- Instant analysis of any construction-related text

### 📊 Analytics Dashboard
- **Key Metrics**: Total notes, sentiment percentages
- **Sentiment Trends**: Interactive time-series charts
- **Distribution Charts**: Pie and bar charts
- **Heatmap**: Daily sentiment calendar view
- **Word Clouds**: Visual representation of common words
- **Date Filtering**: Analyze specific time periods

### 📋 Daily Summary
- AI-generated daily work summaries
- Highlights of positive activities
- Issues requiring attention
- Automated recommendations
- Export functionality

### 📁 File Upload
- Upload custom CSV files
- Automatic sentiment prediction
- Data validation and processing
- Download processed results

## Tech Stack

- **NLP**: scikit-learn, NLTK
- **UI**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data**: Pandas, NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/construction-sentiment-nlp.git
cd construction-sentiment-nlp

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py
```

## Usage

### 1. Generate Sample Data (Optional)
```bash
python src/generate_data.py
```

### 2. Train Model (Optional)
```bash
python src/model.py
```

### 3. Launch Application
```bash
streamlit run src/app.py
```

Navigate to `http://localhost:8501`

## File Upload Format

Your CSV file should have the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| Date | Date (YYYY-MM-DD) | Optional |
| Note | Job note text | Required |
| Sentiment | Positive/Neutral/Negative | Optional |

If the Sentiment column is missing, the app will predict it automatically.

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF (5000 max features)
- **Preprocessing**: Tokenization, Lemmatization, Stopword Removal
- **Accuracy**: 100% on test set (synthetic data)

## Project Structure

```
construction-sentiment-nlp/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── model.py            # Model training script
│   └── generate_data.py    # Dataset generator
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── DEPLOYMENT.md          # Deployment guide
```

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

**Quick Deploy to Streamlit Cloud:**
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

## Screenshots

### Sentiment Analyzer
![Sentiment Analyzer](screenshots/analyzer.png)

### Analytics Dashboard
![Analytics Dashboard](screenshots/dashboard.png)

### Daily Summary
![Daily Summary](screenshots/summary.png)

## Use Cases

- **Project Managers**: Monitor overall project health
- **Site Supervisors**: Quick sentiment checks
- **Executives**: Generate reports for stakeholders
- **Risk Management**: Early warning system for issues

## Future Enhancements

- Multi-language support
- Real-time alerts
- Mobile app version
- Integration with project management tools
- Historical trend predictions

## License

MIT License - feel free to use for your projects!

## Author

Built with ❤️ for construction project management

## Acknowledgments

- Streamlit for the amazing framework
- scikit-learn for ML capabilities
- NLTK for NLP tools
