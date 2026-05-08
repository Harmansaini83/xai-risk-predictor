# Research Paper

## Title
An Explainable AI Framework for Decision Failure Risk
Estimation Using Stacked Ensemble Learning and
Dual Interpretability Methods

---

## Authors
- Harman Saini — Symbiosis Institute of Technology, Nagpur
- Gitika Makheja — Symbiosis Institute of Technology, Nagpur

**Under the guidance of:**
Dr. Gagandeep Kaur — Department of CSE,
Symbiosis Institute of Technology, Nagpur

---

## Status
Submitted for conference review.
Full paper available upon request.

---

## Abstract

The integration of machine learning in credit risk assessment
has improved predictive performance but is limited by poor
transparency, lack of confidence measures, and insufficient
fairness evaluation. This paper proposes an explainable AI
framework combining stacked ensemble learning, dual
interpretability methods, and a reliability scoring mechanism.

The model is evaluated on the Give Me Some Credit dataset
(150,000 records, 93:7 imbalance). A stacked ensemble of
Logistic Regression, Random Forest, XGBoost, and Multilayer
Perceptron is used, with SMOTE for data balancing and
engineered interaction features.

The system achieves 87.96% accuracy and a ROC-AUC of 0.9515.
SHAP provides global explanations while LIME offers local
interpretability. A reliability score (|P − 0.5| × 2)
measures prediction confidence, and a fairness audit across
age groups identifies demographic disparities.

The framework is deployed as a real-time Streamlit application
with chatbot support and automated PDF reporting, demonstrating
a robust and interpretable solution for financial
decision-making.

---

## Key Contributions

**1. Stacked Ensemble Architecture**
Four diverse models combined through a learned meta-learner
achieving 87.96% accuracy — outperforming every individual
model.

**2. Novel Reliability Score**
Formula: Reliability = |P − 0.5| × 2
Converts raw probability to confidence score (0 to 1).
Flags uncertain predictions for human review automatically.
No existing credit scoring system provides this signal.

**3. Dual XAI**
SHAP for global feature attribution + LIME for local
instance-level explanation. Convergence between both methods
validates feature importance rankings scientifically.

**4. Feature Engineering Validation**
Two engineered features (DebtToIncomeRatio and
UtilisationDebtInteraction) confirmed in top-5 by both
Random Forest importance and SHAP independently.

**5. Fairness Audit**
Explicit bias documentation across 5 age demographic groups.
Transparent reporting aligned with responsible AI principles.

**6. End-to-End Deployment**
Complete pipeline from raw CSV to live web application
with downloadable PDF assessment reports.

---

## Keywords

Explainable AI, Credit Risk Assessment, Stacked Ensemble
Learning, SHAP, LIME, Reliability Estimation,
Fairness Audit, Decision Support System, Financial Technology

---

## Paper available upon request

Contact: harmanajitsinghsaini16@gmail.com