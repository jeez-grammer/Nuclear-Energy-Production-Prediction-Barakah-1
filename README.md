# ⚡ Nuclear Energy Production Prediction: Barakah-1

## 📌 Project Overview
This project implements a predictive model to forecast the monthly energy generation (GWh) of the Barakah-1 nuclear reactor (APR-1400) in the United Arab Emirates.

The primary challenge was the **"Maintenance Dip"**—extreme fluctuations in energy output caused by scheduled refueling and maintenance outages. The goal was to build a model capable of accurately predicting both peak operational output and the steep declines associated with these maintenance windows.

---

## 📊 Data Source & Verifiability
The dataset was extracted from official Power Reactor Information System (PRIS) reports provided by the International Atomic Energy Agency (IAEA).

*   **Reactor Model:** APR-1400
*   **Reference Unit Power:** 1,337 MW(e)
*   **Target Variable:** Electricity Produced (GWh)
*   **Key Predictor:** `PUF` (Planned Unavailability Factor) — the percentage of time the plant is offline for scheduled work.
*   **Database Portal:** [IAEA PRIS Analytics Portal (Reactor 1050)](https://iaea.org)

---

## 🛠️ Technical Pipeline

### 1. Data Engineering & Parsing
The source data was provided in a complex, non-tabular format. I developed a custom Python preprocessing pipeline using `io.StringIO` to:
*   Isolate time-series tables from unstructured text blocks.
*   Clean string-formatted numerical data (e.g., converting `"1,012.34"` to float).
*   Engineer a Lag Feature (`Prev_Month_Energy`) to account for operational production momentum.

### 2. Model Iteration: The "Fail & Fix"
I tested multiple architectures to determine the best fit for a high-variance, small-sample dataset:

*   **Attempt 1: Random Forest Regressor** ❌
    *   **Result:** $R^2$ Score: `-187.02` (Extreme Overfitting).
    *   **Diagnosis:** The model was too complex for 12 data points, attempting to "memorize" noise rather than identify the underlying operational pattern.

<img width="844" height="453" alt="Screenshot 2026-08-03 at 1 38 46 PM" src="https://github.com/user-attachments/assets/15d38ded-db43-4e8d-9fc8-1341c6b19593" />

*   **Attempt 2: Ridge Regression (L2 Regularization)** ✔️
    *   **Result:** $R^2$ Score: `0.99` | MAE: `18.49` GWh.
    *   **Diagnosis:** By applying L2 regularization, the model stabilized the coefficients and successfully captured the deterministic relationship between the Planned Unavailability Factor and energy output.

<img width="861" height="452" alt="Screenshot 2026-08-03 at 1 40 13 PM" src="https://github.com/user-attachments/assets/fd9f9e5d-9172-470b-9a2d-4f7db7e1d707" />

---

## 🚀 Final Results & Insights

The final model demonstrates near-perfect alignment with actual production, accurately predicting peaks ($\approx 1,000$ GWh) and maintenance troughs ($\approx 190$ GWh).

| Metric | Value |
| :--- | :--- |
| **$R^2$ Score** | 0.99 |
| **Mean Absolute Error (MAE)** | 18.49 GWh |
| **Winning Model** | Ridge Regression |

### 💡 Domain Insight: The Refueling Cycle
Cross-referencing the model results with the technical report reveals that the production dip in Spring 2025 aligns exactly with the reactor's documented **18-month refueling frequency**. This confirms that the model is capturing a physical operational reality rather than a statistical anomaly.

---

## 💻 Tech Stack
*   **Language:** Python 3.x
*   **Libraries:** `Pandas`, `Scikit-Learn`, `Matplotlib`, `io`

---

## 📈 Future Work
As historical data grows beyond the 2025 window, I plan to:
*   Transition to **Prophet** (by Meta) to model multi-year seasonality.
*   Implement **LSTM** (Long Short-Term Memory) networks to track long-term thermal capacity degradation.

This project demonstrates my ability to handle industrial-grade data, iterate through model failures, and apply domain-specific knowledge to achieve high-accuracy predictive results.

---
**Developed by Nelson Viernes**  
*Database Administrator*
