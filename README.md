# FinOps Cloud Cost Forecasting for EV Telematics Platforms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jclepro-green.svg)](https://doi.org/10.1016/j.jclepro.2025.xxxxx)

## 📄 About

This repository contains the source code and data for the research paper:

> **"Machine Learning-Powered FinOps: Cloud Cost Forecasting and Optimization for Electric Vehicle Telematics Platforms"**
>
> Víctor Valdivieso, Francisco Manuel Arrabal-Campos*, Christian Sonderstrub, Dora Cama-Pinto, Alejandro Cama-Pinto
>
> *Journal of Cleaner Production* (2025)

The study demonstrates how machine learning forecasting methods can be integrated with FinOps (Cloud Financial Operations) practices to achieve substantial cost reductions in cloud-native EV telematics platforms.

## 🎯 Key Results

| Metric | Pre-FinOps | Post-FinOps | Change |
|--------|------------|-------------|--------|
| Vehicle Fleet | 100 | 440 | **+340%** |
| Monthly Messages | 100 | 1,007 | **+907%** |
| Total Cloud Cost | 100 | 45 | **-55%** |
| Cost per Billion Messages | 100 | 4.48 | **-95.5%** |
| **Efficiency Factor** | 1.0× | **22.3×** | **+2,130%** |

## 📊 Forecasting Models Performance

| Model | R² | RMSE | MAE | MAPE (%) | sMAPE (%) |
|-------|-----|------|-----|----------|-----------|
| Polynomial (degree 2) | 0.996 | 1.221 | 1.034 | 2.05 | 2.05 |
| ARIMA (1,1,0) | 0.989 | 1.974 | 1.815 | 3.13 | 3.17 |
| Facebook Prophet | 0.924 | 5.312 | 4.560 | 8.37 | 8.56 |
| LSTM Neural Network | 0.948 | 4.125 | 3.307 | 5.24 | 5.02 |

## 🔬 Methodology

### FinOps Framework
The study applies the **Inform-Optimize-Operate** FinOps framework:

1. **Inform Phase**: Cost visibility, allocation, and benchmarking
2. **Optimize Phase**: Rate optimization, usage optimization, and resource rightsizing
3. **Operate Phase**: Continuous monitoring, governance, and anomaly detection

### Five-Layer Optimization Model
```
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: PURCHASING (Commitment discounts, Reserved)       │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: HARDWARE (Rightsizing, Instance selection)        │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: CONFIGURATION (Autoscaling, Lifecycle policies)   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: IMPLEMENTATION (Query optimization, Caching)      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: ACTIVITY (Data retention, Sampling strategies)    │
└─────────────────────────────────────────────────────────────┘
```

### Machine Learning Forecasting Approaches

1. **Polynomial Regression**: Demand-controlled baseline relating cloud cost to vehicle count
   - Equation: `C(V) = 39.64 - 0.334V + 0.0094V²`
   - Best overall performance (MAPE = 2.05%)

2. **ARIMA (1,1,0)**: Time series forecasting capturing temporal autocorrelation
   - Robust performance (MAPE = 3.13%)

3. **Facebook Prophet**: Automatic trend and seasonality handling
   - Best for business planning scenarios

4. **LSTM Neural Network**: Deep learning for non-linear pattern recognition
   - Potential for improved performance with longer training windows

### Counterfactual Analysis
The polynomial regression model enables counterfactual analysis by projecting what costs *would have been* without FinOps interventions, allowing rigorous attribution of savings.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/finops-ev-telematics.git
cd finops-ev-telematics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run the complete analysis
python src/finops_forecasting.py

# Output files will be generated:
#   - Figure3_column.png (publication figure)
#   - model_metrics_final.csv (performance metrics)
```

## 📁 Repository Structure

```
finops-ev-telematics/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── CITATION.cff              # Citation information
├── .gitignore               # Git ignore rules
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── finops_forecasting.py # Main analysis script
│   ├── models/               # Model implementations
│   │   ├── __init__.py
│   │   ├── polynomial.py
│   │   ├── arima.py
│   │   ├── prophet_model.py
│   │   └── lstm.py
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
│
├── data/                     # Data files
│   ├── README.md
│   └── sample_data.csv
│
├── figures/                  # Generated figures
│   └── README.md
│
├── docs/                     # Documentation
│   ├── methodology.md
│   └── api_reference.md
│
└── tests/                    # Unit tests
    ├── __init__.py
    └── test_models.py
```

## 📈 Generated Figures

### Figure 3: Forecasting Results and FinOps Impact

The main visualization includes three panels:

- **(a) Forecasting Models Comparison**: 40-month predictions from all ML models with 95% prediction intervals
- **(b) Counterfactual Analysis**: Actual costs vs. projected costs without FinOps (orange shading = 95% PI, green shading = savings)
- **(c) Efficiency Evolution**: 22.3× improvement in cost per billion messages

## 🔧 Configuration

### Model Parameters

```python
# Polynomial Regression
POLYNOMIAL_DEGREE = 2

# ARIMA
ARIMA_ORDER = (1, 1, 0)

# LSTM
LSTM_LOOKBACK = 3
LSTM_UNITS = [50, 30]
LSTM_EPOCHS = 300
LSTM_BATCH_SIZE = 2
LSTM_LEARNING_RATE = 0.005

# Prophet
PROPHET_CHANGEPOINT_PRIOR = 0.05
```

### Visualization Settings

```python
# Publication quality settings
FIGURE_DPI = 300
FONT_FAMILY = 'serif'
FONT_SIZE = 11
```

## 📚 Dependencies

- **numpy** >= 1.21.0
- **pandas** >= 1.3.0
- **matplotlib** >= 3.4.0
- **scikit-learn** >= 0.24.0
- **statsmodels** >= 0.12.0
- **tensorflow** >= 2.6.0
- **prophet** >= 1.0.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{valdivieso2025finops,
  title={Machine Learning-Powered FinOps: Cloud Cost Forecasting and Optimization for Electric Vehicle Telematics Platforms},
  author={Valdivieso, Víctor and Arrabal-Campos, Francisco Manuel and Sonderstrub, Christian and Cama-Pinto, Dora and Cama-Pinto, Alejandro},
  journal={Journal of Cleaner Production},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.jclepro.2025.xxxxx}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

The authors gratefully acknowledge **Microsoft EMEA Client Delivery Partner** (Microsoft Campus, Carmenhall Road, Sandyford Business Park, Dublin D18 FW88, Ireland) for their technical support and collaboration in the development and optimization of the Azure cloud infrastructure that enabled this research.

## 📧 Contact

**Francisco Manuel Arrabal-Campos** (Corresponding Author)
- Email: [your-email@university.edu]
- ORCID: [0000-0000-0000-0000]

---

<p align="center">
  <i>Supporting sustainable cloud computing for electric vehicle platforms</i>
</p>
