
# optical-thermometer-ML

> Contactless temperature sensing via polymer birefringence — combining wave optics, experimental physics, and machine learning.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20SVR-green) ![R2](https://img.shields.io/badge/R²-0.93-brightgreen) ![License](https://img.shields.io/badge/license-MIT-orange)

---

## Research question

How can we measure temperature by exploiting the birefringence of adhesive tape — enabling precise, contactless thermal sensing from an inexpensive everyday material?

---

## Overview

This project demonstrates that common adhesive tape — placed between crossed polarisers — acts as a temperature-dependent optical element. As temperature rises, thermal agitation relaxes the polymer chain orientation, reducing the birefringence index Δn. By measuring this optical response and modelling it with machine learning, a functional proof-of-concept contactless thermometer was built and validated.



---

## Key results

| Parameter | Value |
|---|---|
| Birefringence index Δn at 21°C | 9.07 × 10⁻³ |
| Tape thickness e | (42 ± 4) µm |
| Temperature range studied | 25°C – 75°C |
| Random Forest R² score | **0.93** |
| SVR R² score | 0.83 |
| Random Forest MSE | 1.03 × 10⁻⁸ |

---

## Methodology

### 1. Experimental setup
- Polarimetric bench: white light source → linear polariser → tape sample (45°) → analyser (90°) → spectrophotometer
- Channelled spectrum analysis: optical path difference δ = e·Δn extracted via linear regression on fringe order vs inverse wavelength
- Temperature control: water bath (25–75°C) with calibrated probe + photodiode signal acquisition

### 2. Thermal hysteresis
A viscoelastic hysteresis was identified between heating and cooling cycles — an original experimental observation attributed to finite polymer chain relaxation times. Hysteresis-affected datapoints were flagged and excluded from model training.

### 3. Machine learning models
Two supervised regression models were trained and benchmarked on the cleaned experimental dataset:
- **Random Forest** (ensemble of decision trees, bootstrap sampling) — R² = 0.93
- **Support Vector Regressor** with RBF kernel — R² = 0.83

Random Forest significantly outperforms SVR in the non-linear plateau region, with tighter and more homogeneous residuals across the full temperature range.

---

## Tech stack

- Python 3.10
- scikit-learn (Random Forest, SVR, cross-validation, hyperparameter search)
- NumPy, Pandas
- Matplotlib

---

## Repository structure

```
optical-thermometer-ML/
├── data/               # Experimental measurements (temperature, photodiode signal, Δn)
├── notebooks/          # Jupyter notebooks: data cleaning, regression, ML training
├── models/             # Saved Random Forest and SVR models
├── figures/            # Plots: hysteresis, model comparison, error analysis
└── README.md
```

---

## Perspectives

- Extend temperature range below 0°C and above 80°C
- Test different polymer types and tape brands
- Integrate into a compact embedded system for real-time monitoring
- Apply LSTM networks to full time-series signal for dynamic hysteresis handling

---

## Author

**Zainab Ramsis** · Engineering Student · IMT Atlantique  

