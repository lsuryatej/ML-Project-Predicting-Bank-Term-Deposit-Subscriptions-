# Bank Term Deposit Prediction: Synthetic vs Real Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A comprehensive machine learning project demonstrating the effectiveness of synthetic data in financial applications through advanced transfer learning and robust validation methodologies.

## Overview

This project investigates whether high-quality synthetic data can effectively substitute for real customer data in predicting bank term deposit subscriptions. Through rigorous experimentation and validation, we demonstrate **93.7% performance retention** when using synthetic data instead of real data.

### Key Findings

- **Performance Retention**: 93.7% (F1: 0.555 synthetic vs 0.594 real)
- **Data Efficiency**: 91% performance with only 10% of real data
- **Model Discrimination**: ROC-AUC of 0.927 (outstanding)
- **Business Impact**: 5.5x better targeting than random selection

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bank-term-deposit-prediction.git
   cd bank-term-deposit-prediction
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test the installation**
   ```bash
   python main.py
   ```

### Usage

**Run the main application:**
```bash
python main.py
```

**Run the enhanced demo with advanced features:**
```bash
python enhanced_demo.py
```

**Using your own data:**
The project includes sample data files for testing. To use your own datasets, replace the files in `data/raw/`:
- `real.csv` - Real bank marketing dataset (semicolon-separated)
- `synth_train.csv` - Synthetic training data (comma-separated)  
- `synth_test.csv` - Synthetic test data (optional, comma-separated)

**Data Format Requirements:**
- Target column should be named 'y' or configure in `config.yaml`
- Supported formats: CSV with semicolon or comma separators
- Missing values are handled automatically

## Features

### Core Capabilities
- **Interactive Demo**: Complete walkthrough of the ML pipeline with live results
- **Enhanced Demo**: Advanced features including threshold optimization and efficiency analysis
- **Synthetic Data Validation**: Comprehensive comparison of synthetic vs real data performance
- **Transfer Learning**: Advanced techniques for knowledge transfer between datasets
- **Data Efficiency Analysis**: Optimization of data collection requirements
- **Robust Validation**: Proper train/validation/test splits with overfitting prevention

### Supported Models
- Logistic Regression
- Random Forest
- XGBoost (if available)
- LightGBM (if available)
- Neural Networks (optional)

### Advanced Features
- Feature engineering with interaction and ratio features
- Outlier detection and handling
- Class imbalance correction
- Comprehensive performance metrics
- Automatic threshold optimization
- Data efficiency analysis with multiple fractions
- Business impact assessment
- Optimized model configurations

## Project Structure

```
bank-term-deposit-prediction/
├── data/
│   └── raw/                  # Sample datasets (replace with your own)
├── src/                      # Core source code
│   ├── __init__.py
│   ├── data_utils.py         # Data loading & validation utilities
│   ├── enhanced_preprocessing.py    # Advanced preprocessing pipeline
│   └── enhanced_transfer_learning.py # Transfer learning methods
├── artifacts/                # Generated outputs (models, plots, results)
├── main.py                   # Main application entry point
├── enhanced_demo.py          # Enhanced demo with advanced features
├── graph_data_for_report.py  # Visualization data generator
├── config.yaml               # Configuration settings
├── requirements.txt          # Python dependencies
├── report.md                 # Comprehensive analysis report
├── results_dashboard.html    # Interactive visualization dashboard
├── LICENSE                   # MIT license
└── README.md                 # This file
```

## Configuration

The project uses `config.yaml` for configuration. Key settings include:

- **Dataset paths**: Location of your CSV files
- **Model parameters**: Hyperparameters for each algorithm
- **Experiment settings**: Data fractions, validation splits
- **Feature engineering**: Advanced preprocessing options

## Results

### Performance Summary

| Experiment | Model | F1 Score | ROC-AUC | Performance Retention |
|------------|-------|----------|---------|---------------------|
| Real → Real | XGBoost | 0.594 | 0.925 | 100% (baseline) |
| Synthetic → Real | XGBoost | 0.555 | 0.927 | 93.7% |
| Small Real (10%) | XGBoost | 0.539 | 0.900 | 90.7% |

### Key Insights

- Synthetic data achieves 93.7% of real data performance
- Only 10% of real data needed for 91% performance
- Transfer learning shows promising results for data-scarce scenarios
- Robust validation confirms no overfitting

## Methodology

Our approach ensures reliable results through:

- **Proper Data Splits**: Stratified train/validation/test splits
- **Overfitting Prevention**: Validation-test performance gap < 0.01
- **No Data Leakage**: Preprocessing fitted only on training data
- **Conservative Modeling**: Regularized parameters and cross-validation
- **Comprehensive Metrics**: Multiple evaluation approaches

## Documentation

- **[Complete Analysis Report](report.md)**: Detailed methodology and results
- **[Results Dashboard](results_dashboard.html)**: Interactive charts and visualization data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Test your changes: `python main.py`
5. Commit your changes (`git commit -m 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{bank-term-deposit-prediction,
  title={Bank Term Deposit Prediction: Synthetic vs Real Data Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/bank-term-deposit-prediction}
}
```

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) for the Portuguese Bank Marketing Dataset
- [Kaggle Playground Series](https://www.kaggle.com/competitions) for synthetic data generation techniques
- The open-source community for scikit-learn, XGBoost, and LightGBM

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/bank-term-deposit-prediction/issues) page
2. Review the [documentation](report.md) for detailed information
3. Try running `python main.py` to test your setup
4. Open a new issue if your problem isn't already addressed

---

**If you find this project useful, please consider giving it a ⭐!**