# Survival Analysis and Customer Lifetime Value (CLV)

## Overview
This homework performs a full survival analysis and Customer Lifetime Value (CLV) estimation using telecom customer data.
The goal is to identify the factors that influence churn risk, estimate customer profitability, detect at-risk subscribers, and propose data-driven retention strategies.
The analysis is implemented in **R** inside the file `Survival_Analysis_hmw.Rmd`.

---

## Files in This Repository
- **Survival_Analysis_hmw.Rmd** – Main R Markdown file containing all code, plots, and written answers.
- **README.md** – Documentation and instructions for reproducing the analysis.

---

## Data Description
The dataset includes key variables used for survival and CLV modeling:

- `tenure` – Customer lifetime in months  
- `churn` – Churn indicator (1 = churned, 0 = still active)  
- `age` – Customer age  
- `income` – Annual income  
- `gender` – Male/Female  
- `internet` – Whether the customer subscribes to Internet service  
- `forward` – Whether the customer uses Forwarding features  
- `custcat` – Customer category (Basic, E-service, Plus service, Total service)

These features are used to model time-to-churn and predict long-term customer value.

---

## Methods

### 1. Survival Preparation
- The survival object is constructed using  
  `Surv(tenure, churn)`.
- Variables are cleaned and converted to factors as needed.

### 2. Parametric AFT Models
The following Accelerated Failure Time (AFT) models were estimated:

- Exponential  
- Weibull  
- Lognormal  
- Log-logistic  

Models were compared using AIC.  
**The Lognormal distribution was selected as the best fit.**

### 3. Coefficient Interpretation
The model identifies how each predictor affects churn:

- Higher age and income → lower churn risk  
- Internet and Forwarding usage → longer predicted retention  
- Premium categories (E-service, Plus, Total) → significantly extended survival time  
- Gender: males show slightly lower churn risk, but females generate higher CLV due to higher average margins  

### 4. CLV Estimation
- Monthly margin: **$12**  
- Monthly discount rate: **1%**  
- Survival probabilities for each customer are predicted from the Lognormal AFT model.  
- CLV is computed as the discounted sum of expected future margins.  
- CLV distribution typically falls between \$400–\$600, with premium-tier customers showing the highest values.

### 5. Most Valuable Segments
CLV segmentation reveals:

- **Plus**, **E-service**, and **Total service** users are the most profitable.
- Higher-income and senior-age customers also have high CLV.
- Female customers generate higher CLV despite males having slightly longer predicted survival.

### 6. At-Risk Customers & Retention Budget
- At-risk threshold: **S(12) < 0.60**  
- At-risk customers: **27** (2.7% of population)  
- Combined 1-year CLV of at-risk group: **\$2,555.49**  
- Retention cost per customer: **\$100**  
- Annual retention budget needed: **\$2,700**  
- ROI: **–5.4%**, meaning high-cost retention is **not** justified for this group.

### 7. Recommendations
- Focus retention efforts on premium-tier, higher-income, and older customers.
- Promote service bundles to increase retention and CLV.
- Use low-cost automated messages for low-value at-risk customers.
- Integrate churn risk (survival probabilities) and CLV into CRM for better targeting.


