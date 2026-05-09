import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import lime
import lime.lime_tabular
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="XAI Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_models():
    model = joblib.load("model_stacked_ensemble.pkl")
    scaler = joblib.load("model_scaler.pkl")
    xgb_model = joblib.load("model_xgboost.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, xgb_model, feature_names

model, scaler, xgb_model, feature_names = load_models()

def get_risk_level(p):
    if p < 0.4:
        return "Low", "🟢"
    elif p < 0.7:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

def get_reliability(p):
    return round(abs(p - 0.5) * 2, 4)

def predict(input_data):
    df_input = pd.DataFrame([input_data], columns=feature_names)
    scaled = scaler.transform(df_input)
    prob = model.predict_proba(scaled)[0][1]
    risk, icon = get_risk_level(prob)
    rel = get_reliability(prob)
    return prob, risk, icon, rel, scaled

st.sidebar.title("XAI Risk Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.markdown("**Domain:** Finance - Credit Risk")
st.sidebar.markdown("**Dataset:** Give Me Some Credit")
st.sidebar.markdown("**Samples:** 150,000 records")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("Accuracy", "87.96%")
st.sidebar.metric("ROC-AUC (Test)", "0.8106")
st.sidebar.metric("ROC-AUC (CV)", "0.9515")
st.sidebar.metric("Mean Reliability", "80.83%")
st.sidebar.markdown("---")
st.sidebar.markdown("### Models Used")
st.sidebar.markdown("Logistic Regression")
st.sidebar.markdown("Random Forest")
st.sidebar.markdown("XGBoost")
st.sidebar.markdown("MLP Neural Network")
st.sidebar.markdown("Stacked Ensemble")
st.sidebar.markdown("---")
st.sidebar.markdown("### XAI Methods")
st.sidebar.markdown("SHAP - Global Explanation")
st.sidebar.markdown("LIME - Local Explanation")
st.sidebar.markdown("---")
domain = st.sidebar.selectbox(
    "Select Domain",
    ["Finance (Active)",
     "Healthcare (Coming Soon)",
     "HR (Coming Soon)",
     "Operations (Coming Soon)"]
)

st.title("Explainable AI - Decision Failure Risk Predictor")
st.markdown(
    "Enter applicant financial details. The system predicts **failure risk**, "
    "explains **why** using SHAP and LIME, and shows a **reliability score**."
)
st.markdown("---")

st.header("Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Credit Behaviour**")
    revolving_util = st.slider(
        "Revolving Utilisation (%)",
        0.0, 1.0, 0.3, 0.01,
        help="Credit card balance divided by credit limit"
    )
    num_30_59 = st.number_input(
        "Times 30-59 Days Late",
        min_value=0, max_value=20, value=0
    )
    num_60_89 = st.number_input(
        "Times 60-89 Days Late",
        min_value=0, max_value=20, value=0
    )
    num_90_days = st.number_input(
        "Times 90+ Days Late",
        min_value=0, max_value=20, value=0
    )

with col2:
    st.markdown("**Financial Info**")
    monthly_income = st.number_input(
        "Monthly Income ($)",
        min_value=0, max_value=100000, value=5000
    )
    debt_ratio = st.slider(
        "Debt Ratio",
        0.0, 1.0, 0.3, 0.01,
        help="Monthly debt payments divided by monthly income"
    )
    open_credit = st.number_input(
        "Open Credit Lines",
        min_value=0, max_value=50, value=5
    )
    real_estate = st.number_input(
        "Real Estate Loans",
        min_value=0, max_value=20, value=1
    )

with col3:
    st.markdown("**Personal Info**")
    age = st.number_input(
        "Age",
        min_value=18, max_value=100, value=45
    )
    dependents = st.number_input(
        "Number of Dependents",
        min_value=0, max_value=20, value=0
    )
    st.markdown("---")
    st.markdown("**Auto-computed Features**")
    debt_to_income = debt_ratio / (monthly_income + 1)
    util_debt = revolving_util * debt_ratio
    st.info(f"Debt-to-Income Ratio: {debt_to_income:.5f}")
    st.info(f"Utilisation x Debt: {util_debt:.5f}")

input_data = [
    revolving_util, age, num_30_59, debt_ratio,
    monthly_income, open_credit, num_90_days,
    real_estate, num_60_89, dependents,
    debt_to_income, util_debt
]

st.markdown("---")
predict_btn = st.button(
    "Analyse Risk Now",
    use_container_width=True
)

if predict_btn:
    if domain != "Finance (Active)":
        st.warning(
            "This domain is coming soon. "
            "Please select Finance (Active)."
        )
    else:
        prob, risk, icon, rel, scaled_input = predict(input_data)

        st.markdown("---")
        st.header("Prediction Dashboard")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Level", f"{icon} {risk}")
        m2.metric("Probability", f"{prob:.4f}")
        m3.metric("Reliability Score", f"{rel:.4f}")
        m4.metric("Confidence", f"{rel*100:.1f}%")

        st.markdown("---")

        col_g, col_r = st.columns(2)

        with col_g:
            st.subheader("Risk Probability Gauge")
            fig_g, ax_g = plt.subplots(figsize=(7, 4))
            ax_g.barh([''], [0.4], color='#2ECC71',
                      alpha=0.25, height=0.5, label='Low zone')
            ax_g.barh([''], [0.3], left=0.4, color='#F39C12',
                      alpha=0.25, height=0.5, label='Medium zone')
            ax_g.barh([''], [0.3], left=0.7, color='#E74C3C',
                      alpha=0.25, height=0.5, label='High zone')
            bar_color = ('#2ECC71' if prob < 0.4
                         else '#F39C12' if prob < 0.7
                         else '#E74C3C')
            ax_g.barh([''], [prob], color=bar_color,
                      height=0.25, alpha=0.95)
            ax_g.axvline(x=prob, color='black',
                         linewidth=2.5, linestyle='--')
            ax_g.text(prob, 0.18, f'{prob:.4f}',
                      ha='center', fontweight='bold', fontsize=13)
            ax_g.set_xlim(0, 1)
            ax_g.set_xlabel('Risk Probability Score')
            ax_g.text(0.20, -0.38, 'LOW', ha='center',
                      color='#2ECC71', fontweight='bold', fontsize=11)
            ax_g.text(0.55, -0.38, 'MEDIUM', ha='center',
                      color='#F39C12', fontweight='bold', fontsize=11)
            ax_g.text(0.85, -0.38, 'HIGH', ha='center',
                      color='#E74C3C', fontweight='bold', fontsize=11)
            ax_g.set_title(
                f'Risk Level: {icon} {risk}  |  Probability: {prob:.4f}',
                fontweight='bold', fontsize=12
            )
            ax_g.legend(fontsize=9, loc='upper left')
            ax_g.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close()

        with col_r:
            st.subheader("Reliability Score Breakdown")
            fig_r, ax_r = plt.subplots(figsize=(7, 4))
            values = [rel, 1 - rel]
            labels_r = [
                f'Reliability\n{rel*100:.1f}%',
                f'Uncertainty\n{(1-rel)*100:.1f}%'
            ]
            colors_r = ['#3498DB', '#ECF0F1']
            wedges, texts, autotexts = ax_r.pie(
                values,
                labels=labels_r,
                colors=colors_r,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 11},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            for at in autotexts:
                at.set_fontweight('bold')
            ax_r.set_title(
                f'Reliability: {rel:.4f}\nFormula: |{prob:.4f} - 0.5| x 2',
                fontweight='bold', fontsize=11
            )
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close()

        st.markdown("---")
        st.subheader("AI Explanation - Plain English")

        risk_color = (
            "green" if risk == "Low"
            else "orange" if risk == "Medium"
            else "red"
        )
        confidence_text = (
            "very high confidence" if rel > 0.8
            else "moderate confidence" if rel > 0.5
            else "low confidence - borderline case"
        )
        risk_advice = (
            "This applicant appears financially stable. "
            "Standard loan processing recommended."
            if risk == "Low"
            else "This applicant shows some risk indicators. "
            "Additional verification recommended."
            if risk == "Medium"
            else "This applicant shows significant risk indicators. "
            "Careful review strongly recommended."
        )

        st.markdown(f"""
        > **Risk Assessment Summary:**
        >
        > The stacked ensemble model predicts a
        > **:{risk_color}[{risk} Risk]** level with a probability
        > of **{prob:.4f}**.
        > The system has **{confidence_text}**
        > in this prediction (Reliability Score: {rel:.4f}).
        >
        > **Recommendation:** {risk_advice}
        >
        > A reliability of {rel:.4f} means the model is
        > **{rel*100:.1f}% certain** about this decision.
        """)

        st.markdown("---")
        st.subheader("SHAP - Global Feature Explanation")
        st.caption(
            "Shows which features pushed the prediction "
            "up or down for this applicant"
        )

        try:
            shap_explainer = shap.TreeExplainer(xgb_model)
            shap_vals = shap_explainer.shap_values(scaled_input)
            fig_s, ax_s = plt.subplots(figsize=(10, 6))
            shap_series = pd.Series(
                shap_vals[0],
                index=feature_names
            ).sort_values(key=abs, ascending=True)
            colors_s = ['#E74C3C' if v > 0
                        else '#3498DB'
                        for v in shap_series.values]
            ax_s.barh(
                shap_series.index,
                shap_series.values,
                color=colors_s,
                edgecolor='white',
                linewidth=0.5
            )
            for i, v in enumerate(shap_series.values):
                xpos = v + 0.01 if v >= 0 else v - 0.01
                ha = 'left' if v >= 0 else 'right'
                ax_s.text(xpos, i, f'{v:+.3f}',
                          va='center', ha=ha,
                          fontsize=9, fontweight='bold')
            ax_s.axvline(x=0, color='black', linewidth=1.5)
            ax_s.set_title(
                'SHAP Feature Contributions - This Applicant\n'
                '(Red = increases risk  |  Blue = decreases risk)',
                fontweight='bold', fontsize=12
            )
            ax_s.set_xlabel('SHAP Value - Impact on Prediction')
            ax_s.legend(handles=[
                mpatches.Patch(color='#E74C3C', label='Increases risk'),
                mpatches.Patch(color='#3498DB', label='Decreases risk')
            ], fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close()

            top3 = shap_series.abs().sort_values(
                ascending=False
            ).head(3).index.tolist()
            st.markdown("**Top 3 factors driving this prediction:**")
            for i, feat in enumerate(top3, 1):
                val = shap_series[feat]
                direc = "increases" if val > 0 else "decreases"
                st.markdown(
                    f"**{i}.** `{feat}` - "
                    f"{direc} risk (impact: {val:+.4f})"
                )
        except Exception as e:
            st.warning(f"SHAP error: {e}")

        st.markdown("---")
        st.subheader("LIME - Local Explanation")
        st.caption(
            "Explains why this specific applicant "
            "received this prediction"
        )

        try:
            dummy_train = np.zeros((100, len(feature_names)))
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=dummy_train,
                feature_names=feature_names,
                class_names=['No Risk', 'At Risk'],
                mode='classification',
                random_state=42
            )
            lime_result = lime_explainer.explain_instance(
                data_row=scaled_input[0],
                predict_fn=model.predict_proba,
                num_features=10,
                num_samples=500
            )
            lime_list = lime_result.as_list()
            lime_names = [x[0] for x in lime_list]
            lime_weights = [x[1] for x in lime_list]
            lime_colors = ['#E74C3C' if w > 0
                           else '#3498DB'
                           for w in lime_weights]
            fig_l, ax_l = plt.subplots(figsize=(10, 6))
            ax_l.barh(
                lime_names, lime_weights,
                color=lime_colors,
                edgecolor='white', linewidth=0.5
            )
            for i, w in enumerate(lime_weights):
                xpos = w + 0.001 if w >= 0 else w - 0.001
                ha = 'left' if w >= 0 else 'right'
                ax_l.text(xpos, i, f'{w:+.4f}',
                          va='center', ha=ha,
                          fontsize=9, fontweight='bold')
            ax_l.axvline(x=0, color='black', linewidth=1.5)
            ax_l.set_title(
                'LIME Local Explanation - This Specific Applicant\n'
                '(Red = risk factor  |  Blue = protective factor)',
                fontweight='bold', fontsize=12
            )
            ax_l.set_xlabel('Feature Weight')
            ax_l.legend(handles=[
                mpatches.Patch(color='#E74C3C', label='Risk factor'),
                mpatches.Patch(color='#3498DB', label='Protective factor')
            ], fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_l)
            plt.close()
        except Exception as e:
            st.warning(f"LIME error: {e}")

        st.markdown("---")
        st.subheader("Download Full Report")

        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 18)
            pdf.cell(0, 12,
                     "XAI Decision Risk Assessment Report",
                     ln=True, align='C')
            pdf.set_font("Arial", size=11)
            pdf.cell(0, 8,
                     "Explainable AI Framework - Finance Domain",
                     ln=True, align='C')
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 13)
            pdf.set_fill_color(44, 62, 80)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, "  PREDICTION RESULT",
                     ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=12)
            pdf.ln(3)
            pdf.cell(0, 8, f"  Risk Level        : {risk}",
                     ln=True)
            pdf.cell(0, 8, f"  Probability       : {prob:.4f}",
                     ln=True)
            pdf.cell(0, 8, f"  Reliability Score : {rel:.4f}",
                     ln=True)
            pdf.cell(0, 8, f"  Confidence        : {rel*100:.1f}%",
                     ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 13)
            pdf.set_fill_color(44, 62, 80)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, "  MODEL INFORMATION",
                     ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=11)
            pdf.ln(3)
            pdf.cell(0, 7,
                     "  Model    : Stacked Ensemble"
                     " (LR + RF + XGBoost + MLP)",
                     ln=True)
            pdf.cell(0, 7, "  Accuracy : 87.96%", ln=True)
            pdf.cell(0, 7,
                     "  ROC-AUC  : 0.8106 (Test)"
                     " / 0.9515 (Cross-Validation)",
                     ln=True)
            pdf.cell(0, 7,
                     "  XAI      : SHAP + LIME"
                     " (Dual Explainability)",
                     ln=True)
            pdf.cell(0, 7,
                     "  Dataset  : Give Me Some Credit"
                     " (150,000 records)",
                     ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 13)
            pdf.set_fill_color(44, 62, 80)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, "  APPLICANT INPUT DATA",
                     ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=10)
            pdf.ln(3)
            for fname, fval in zip(feature_names, input_data):
                safe_name = str(fname).encode(
                    'ascii', 'ignore').decode('ascii')
                safe_val = str(round(float(fval), 5))
                pdf.cell(0, 7,
                         f"  {safe_name}: {safe_val}",
                         ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 13)
            pdf.set_fill_color(44, 62, 80)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 10, "  RECOMMENDATION",
                     ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", size=11)
            pdf.ln(3)
            safe_advice = str(risk_advice).encode(
                'ascii', 'ignore').decode('ascii')
            pdf.multi_cell(0, 8, f"  {safe_advice}")
            pdf.ln(5)

            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8,
                     "Generated by XAI Decision Failure"
                     " Risk Framework - BTech CSE Project",
                     ln=True, align='C')

            pdf_bytes = bytes(pdf.output())
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="risk_assessment_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("PDF ready. Click above to download.")

        except Exception as e:
            st.error(f"PDF error: {e}")

st.markdown("---")
st.markdown(
    "<center><small>"
    "XAI Decision Failure Risk Framework - "
    "By Harman Saini , SIT Nagpur PBL Sem4 - "
    "Stacked Ensemble + SHAP + LIME + Reliability Scoring"
    "</small></center>",
    unsafe_allow_html=True
)