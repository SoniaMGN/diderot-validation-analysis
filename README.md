# Empirical Validation of the Diderot Effect in Beauty Consumption

This repository contains the data pipeline and statistical analysis for a thesis investigating the **Diderot Effect** within the Amazon "All Beauty" category. The project uses Natural Language Processing (NLP) and purchase history modeling to prove that aesthetic triggers lead to a "spending spiral."

## Executive Summary of Findings
* **Hypothesis 1 (Linguistic Shift):** Confirmed. After an aesthetic trigger, users show a **36.7% increase** in congruence-related language (e.g., "matching," "set," "harmony").
* **Hypothesis 2 (Temporal Acceleration):** Confirmed. The inter-purchase interval ($\Delta T$) dropped from **75.6 days** to **38.8 days** using data-driven change point detection — a **49% acceleration**.
* **Predictive Link:** Cox Proportional Hazards model confirms that being in the Congruence phase increases purchase hazard by **24%** (HR = 1.24, $p < 0.0001$), and that lagged aesthetic score independently accelerates purchasing (HR = 1.07, $p < 0.0001$).

---

## Project Structure & File Descriptions

### 1. Data Processing & Phase Identification
* **`identify_triggers.py`**: The entry point of the pipeline. It standardizes timestamps and applies a "Phase Labeling" logic. It identifies the first instance where a user's review exceeds an aesthetic threshold ($S > 0.6$) and splits their history into **Utilitarian** (pre-trigger) and **Congruence** (post-trigger) phases.
* **`calculate_velocity.py`**: Calculates the temporal distance between purchases. It groups data by `user_id` and uses a diffing function to find the number of days ($\Delta T$) between Review $N$ and Review $N-1$.

### 2. Linguistic Analysis
* **`content_analysis.py`**: Performs Automated Content Analysis. It uses regular expressions to count "Utilitarian" vs. "Congruence" markers in review text and aggregates the mean scores for the final thesis table.
* **Word Embeddings (Logic)**: The dataset incorporates **Mixedbread-ai Transformer embeddings** to calculate the **Aesthetic Score ($S$)**, measuring the semantic similarity of reviews to a "Minimalist/Clean-Girl" aesthetic anchor.

### 3. Statistical Validation
* **`changepoint_analysis.py`**: Runs **Bayesian Change Point Detection** (ruptures Pelt algorithm) on each user's aesthetic score time series. Instead of using a hardcoded threshold, it lets the data determine where each user's consumption regime shifts. Outputs data-driven Utilitarian/Congruence phase labels and a comparison against the threshold method.
* **`survival_analysis.py`**: Runs a **Cox Proportional Hazards model** (lifelines). Models the *hazard* of making the next purchase as a function of lagged linguistic features and aesthetic score. Produces Kaplan-Meier survival curves and a forest plot of hazard ratios.

### 4. Visualization & Reporting
* **`diderot_analysis.py`**: Generates the primary thesis charts — Temporal Acceleration, Linguistic Shift, and Aesthetic Score Shift — as a multi-page PDF.
* **`diderot_thesis_results.pdf`**: Consolidated PDF with all major visualizations.
* **`changepoint_report.pdf`**: Phase distribution comparison, method agreement analysis, and example user regime change plots.
* **`survival_report.pdf`**: Kaplan-Meier curves, Cox hazard ratio forest plot, and model summary.

---

##  Core Data Metrics
| Metric | Utilitarian Phase | Congruence Phase | Change |
| :--- | :--- | :--- | :--- |
| **Purchase Interval ($\Delta T$)** | 75.6 Days | 38.8 Days | **-49%** |
| **Congruence Markers** | 0.147 | 0.201 | **+36.7%** |
| **Aesthetic Score ($S$)** | 0.521 | 0.566 | **+8.6%** |
| **Purchase Hazard (Cox HR)** | 1.00 (ref) | 1.24 | **+24%** |

> Phase intervals derived from data-driven change point detection (ruptures Pelt, pen=1) rather than a fixed threshold.

---

##  Technical Stack
* **Language:** Python 3.13
* **Data Manipulation:** `Pandas`, `NumPy`
* **Change Point Detection:** `ruptures`
* **Survival Analysis:** `lifelines` (Cox PH, Kaplan-Meier)
* **NLP:** `re` (Regex), `spaCy`, `Sentence-Transformers`
* **Visualization:** `Matplotlib`, `Seaborn`

## Methodology Note
This project uses two complementary approaches to establish directionality. **Bayesian Change Point Detection** (ruptures) identifies each user's regime shift from the data itself, without relying on a hardcoded aesthetic score threshold. **Cox Proportional Hazards** modelling then tests whether lagged linguistic features at $T_{n-1}$ predict the purchase hazard at $T_{n}$, confirming that the psychological shift toward "Congruence" acts as a leading indicator of accelerated consumption.

## Environment Setup

Copy `.env.example` to `.env` and add your Mixedbread API key:

```
MIXEDBREAD_API_KEY=your_key_here
```


## 💻 How to Run the Analysis
To replicate the results from the 15,000-row beauty dataset:

1. **Clone the repo:** `https://github.com/SoniaMGN/diderot-validation-analysis`

2. **Setup Environment:**
   ```bash
   python -m venv diderot_venv
   # Windows:
   diderot_venv\Scripts\activate
   # macOS/Linux:
   # source diderot_venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API Key:** Copy `.env.example` to `.env` and set `MIXEDBREAD_API_KEY=your_key_here`

4. **Run the full pipeline:**
   ```bash
   cd analysis
   python pipeline.py
   ```

   Or run individual stages:
   ```bash
   python pipeline.py --stage embed       # embeddings only
   python pipeline.py --from-stage 4     # resume from stage 4
   ```

### Pipeline Stages
| # | Stage | Script | Output |
|---|-------|--------|--------|
| 1 | Load & filter | `initial_data_analysis.py` | `diderot_working_set.jsonl` |
| 2 | Category sort | `diderot_sequential_sorting.py` | `diderot_final_sequence.jsonl` |
| 3 | Embeddings | `embeddings.py` | `diderot_final_scored.jsonl` |
| 4 | Phase labeling | `identify_triggers.py` | `diderot_with_triggers.csv` |
| 5 | Velocity | `calculate_velocity.py` | `diderot_with_velocity.csv` |
| 6 | Content analysis | `content_analysis.py` | `final_diderot_report.csv` |
| 7 | Change point detection | `changepoint_analysis.py` | `diderot_changepoint.csv`, `changepoint_report.pdf` |
| 8 | Survival analysis | `survival_analysis.py` | `survival_report.pdf` |
| 9 | Visualizations | `diderot_analysis.py` | `diderot_thesis_results.pdf` |