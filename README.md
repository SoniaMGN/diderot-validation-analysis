# Empirical Validation of the Diderot Effect in Beauty Consumption

This repository contains the data pipeline and statistical analysis for a thesis investigating the **Diderot Effect** within the Amazon "All Beauty" category. The project uses Natural Language Processing (NLP) and purchase history modeling to prove that aesthetic triggers lead to a "spending spiral."

## Executive Summary of Findings
* **Hypothesis 1 (Linguistic Shift):** Confirmed. After an aesthetic trigger, users show a **36.7% increase** in congruence-related language (e.g., "matching," "set," "harmony").
* **Hypothesis 2 (Temporal Acceleration):** Confirmed. The inter-purchase interval ($\Delta T$) dropped from **144.14 days** to **82.17 days**.
* **Predictive Link:** Lagged OLS Regression proves that aesthetic language is a statistically significant predictor ($p = 0.028$) of faster subsequent purchases.

---

## Project Structure & File Descriptions

### 1. Data Processing & Phase Identification
* **`identify_triggers.py`**: The entry point of the pipeline. It standardizes timestamps and applies a "Phase Labeling" logic. It identifies the first instance where a user's review exceeds an aesthetic threshold ($S > 0.6$) and splits their history into **Utilitarian** (pre-trigger) and **Congruence** (post-trigger) phases.
* **`calculate_velocity.py`**: Calculates the temporal distance between purchases. It groups data by `user_id` and uses a diffing function to find the number of days ($\Delta T$) between Review $N$ and Review $N-1$.

### 2. Linguistic Analysis
* **`content_analysis.py`**: Performs Automated Content Analysis. It uses regular expressions to count "Utilitarian" vs. "Congruence" markers in review text and aggregates the mean scores for the final thesis table.
* **Word Embeddings (Logic)**: The dataset incorporates **Mixedbread-ai Transformer embeddings** to calculate the **Aesthetic Score ($S$)**, measuring the semantic similarity of reviews to a "Minimalist/Clean-Girl" aesthetic anchor.

### 3. Statistical Validation
* **`correlation_analysis.py`**: Runs a **Lagged Variable OLS Regression**. It tests if the language used in a previous purchase predicts the speed of the next purchase. This file generates the statistical proof for the "Diderot Spiral."

### 4. Visualization & Reporting
* **`visualize_diderot_results.py`**: Generates the primary charts for the thesis, including the Temporal Acceleration bar chart and the Linguistic Shift grouped bars.
* [cite_start]**`diderot_thesis_results.pdf`**: A consolidated PDF report containing all major visualizations and the final summary table.
* **`diderot_correlation_report.pdf`**: The formal output of the OLS Regression, showing coefficients, P-values, and significance levels.

---

##  Core Data Metrics
| Metric | Utilitarian Phase | Congruence Phase | Change |
| :--- | :--- | :--- | :--- |
| **Purchase Interval ($\Delta T$)** | 144.14 Days  | 82.17 Days | **-43.0%** |
| **Congruence Markers** | 0.147 | 0.201 | **+36.7%** |
| **Aesthetic Score ($S$)** | 0.521  | 0.566  | **+8.6%** |

---

##  Technical Stack
* **Language:** Python 3.13
* **Data Manipulation:** `Pandas`, `NumPy`
* **Statistical Modeling:** `Statsmodels` (OLS Regression)
* **NLP:** `re` (Regex), `spaCy`, `Sentence-Transformers`
* **Visualization:** `Matplotlib`, `Seaborn`

## Methodology Note
This project utilizes **Lagged Variable Analysis** to ensure directionality. By analyzing if language at $T_{n-1}$ predicts behavior at $T_{n}$, we establish that the psychological shift toward "Congruence" acts as a leading indicator for the increase in consumption velocity.

# .env.example - Copy this to .env and add your keys
MIXEDBREAD_API_KEY=your_key_here


## 💻 How to Run the Analysis
To replicate the results from the 15,000-row beauty dataset:

1. **Clone the repo:** `https://github.com/SoniaMGN/diderot-validation-analysis`
2. **Setup Environment:** ```bash
   python3 -m venv diderot_venv
   source diderot_venv/bin/activate
   pip install -r requirements.txt