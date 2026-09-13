import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import io

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Barrier // Customer Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling & Design System
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .main {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

code, kbd, pre, samp {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide excess default Streamlit decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Glassmorphic Brand Banner */
.brand-header {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.brand-title {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(to right, #60a5fa, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.brand-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    font-weight: 400;
}

/* Metric KPI Cards */
.kpi-card {
    background: rgba(18, 24, 38, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(59, 130, 246, 0.4);
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f8fafc;
}

.kpi-sub {
    font-size: 0.76rem;
    color: #64748b;
    margin-top: 0.35rem;
}

/* Prediction Result Card */
.result-card-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(30, 15, 20, 0.7) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 30px rgba(239, 68, 68, 0.2);
}

.result-card-med {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(35, 25, 15, 0.7) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 30px rgba(245, 158, 11, 0.2);
}

.result-card-low {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(15, 30, 20, 0.7) 100%);
    border: 1px solid rgba(34, 197, 94, 0.4);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 30px rgba(34, 197, 94, 0.2);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
.badge-amber { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
.badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
.badge-blue { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }

/* Card Wrap for charts */
.chart-container {
    background: rgba(18, 24, 38, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.25rem 0.75rem;
    margin-bottom: 1.25rem;
}

.chart-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.2rem;
}

.chart-desc {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 0.8rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Plotly Theme Defaults
# ---------------------------------------------------------
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=11),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=10, color="#94a3b8"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.05)",
        tickfont=dict(size=10, color="#94a3b8"),
    ),
)

# ---------------------------------------------------------
# Data & Model Helpers
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    'ord__Contract',
    'ord__Tenure_Group',
    'nom__PhoneService_Yes',
    'nom__PaperlessBilling_Yes',
    'nom__PaymentMethod_Credit card (automatic)',
    'nom__PaymentMethod_Electronic check',
    'nom__PaymentMethod_Mailed check',
    'remainder__tenure',
    'remainder__MonthlyCharges',
    'remainder__TotalCharges'
]

@st.cache_data
def load_data():
    raw_path = "data/churn_data.csv"
    if not os.path.exists(raw_path):
        raw_path = "churn_data.csv"
    df = pd.read_csv(raw_path)
    df['TotalCharges_Clean'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

@st.cache_data
def load_benchmarks():
    path = "artifacts/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def transform_customer_features(df_input):
    """
    Transforms raw customer attributes into the exact 10 features expected by the model.
    """
    d = df_input.copy()
    contract_map = {'Month-to-month': 0.0, 'One year': 1.0, 'Two year': 2.0}
    tenure_group = pd.cut(d['tenure'], bins=[-1, 6, 24, np.inf], labels=['0-6 months', '7-24 months', '25+ months'])
    tenure_group_map = {'0-6 months': 0.0, '7-24 months': 1.0, '25+ months': 2.0}
    
    out = pd.DataFrame()
    out['ord__Contract'] = d['Contract'].map(contract_map).fillna(0.0).astype(float)
    out['ord__Tenure_Group'] = tenure_group.map(tenure_group_map).fillna(0.0).astype(float)
    out['nom__PhoneService_Yes'] = (d['PhoneService'] == 'Yes').astype(float)
    out['nom__PaperlessBilling_Yes'] = (d['PaperlessBilling'] == 'Yes').astype(float)
    out['nom__PaymentMethod_Credit card (automatic)'] = (d['PaymentMethod'] == 'Credit card (automatic)').astype(float)
    out['nom__PaymentMethod_Electronic check'] = (d['PaymentMethod'] == 'Electronic check').astype(float)
    out['nom__PaymentMethod_Mailed check'] = (d['PaymentMethod'] == 'Mailed check').astype(float)
    out['remainder__tenure'] = d['tenure'].astype(float)
    out['remainder__MonthlyCharges'] = d['MonthlyCharges'].astype(float)
    out['remainder__TotalCharges'] = pd.to_numeric(d['TotalCharges'], errors='coerce').fillna(0.0).astype(float)
    return out[FEATURE_COLUMNS]

@st.cache_resource
def load_or_train_models():
    """
    Loads saved model package if available, and ensures reliable RandomForest & XGBoost predictors.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    
    model_pkg = None
    pkg_path = "models/churn_model_package.pkl"
    if os.path.exists(pkg_path):
        try:
            model_pkg = joblib.load(pkg_path)
        except Exception:
            model_pkg = None

    # Load processed data to ensure trained models and scaler match
    proc_path = "data/processed_churn_data.csv"
    if os.path.exists(proc_path):
        proc_df = pd.read_csv(proc_path)
    else:
        raw_df = load_data()
        proc_df = transform_customer_features(raw_df)
        proc_df['remainder__Churn'] = raw_df['Churn']

    proc_df['remainder__TotalCharges'] = pd.to_numeric(proc_df['remainder__TotalCharges'], errors='coerce').fillna(0)
    y = proc_df['remainder__Churn'].map({'No': 0, 'Yes': 1})
    X = proc_df[FEATURE_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    rf_model = RandomForestClassifier(
        class_weight='balanced',
        max_depth=8,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )
    rf_model.fit(x_train_scaled, y_train)

    xgb_model = XGBClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=200,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(x_train_scaled, y_train)

    return {
        'scaler': scaler,
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'x_test': x_test,
        'x_test_scaled': x_test_scaled,
        'y_test': y_test,
        'feature_names': FEATURE_COLUMNS
    }

models_dict = load_or_train_models()
df_raw = load_data()
df_benchmarks = load_benchmarks()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/shield.png", width=64)
    st.markdown("### **Barrier Engine**")
    st.caption("NTI Graduation Project // Barrier-")
    st.divider()

    selected_model_type = st.selectbox(
        "🧠 Active Prediction Model",
        ["Random Forest (Top Performer)", "XGBoost (Gradient Boosted)"],
        index=0,
        help="Select which machine learning classifier computes probabilities."
    )
    active_clf = models_dict['rf_model'] if "Random Forest" in selected_model_type else models_dict['xgb_model']
    
    st.divider()
    st.markdown("#### **Project Quick Links**")
    st.markdown("🔗 [GitHub Repository](https://github.com/Mostafa-Ashraf-Elshahawy/Barrier-)")
    st.markdown("📄 [Customer Dataset (7,042)](https://github.com/Mostafa-Ashraf-Elshahawy/Barrier-/blob/main/data/churn_data.csv)")
    
    st.divider()
    st.markdown("#### **Dataset Summary**")
    total_cust = len(df_raw)
    churn_count = (df_raw['Churn'] == 'Yes').sum()
    churn_rate = (churn_count / total_cust) * 100
    st.metric("Total Customers", f"{total_cust:,}")
    st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 class="brand-title">BARRIER // Churn Intelligence System</h1>
            <div class="brand-subtitle">
                Advanced AI-Powered Customer Retention Platform • Risk Scoring • Retention Simulation
            </div>
        </div>
        <div>
            <span class="badge badge-blue">v1.2 Production</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_pred, tab_batch, tab_eda, tab_benchmarks, tab_about = st.tabs([
    "🔮 Real-Time Risk Predictor",
    "📁 Batch Customer Scoring",
    "📊 Exploratory Data Analysis",
    "🤖 Model Benchmarks & Explainability",
    "ℹ️ Architecture & Documentation"
])

# =========================================================
# TAB 1: Real-Time Risk Predictor & What-If Simulator
# =========================================================
with tab_pred:
    st.markdown("### **Customer Churn Risk Assessment & Retention Simulator**")
    st.caption("Input customer contract and billing profile below, or choose a pre-configured persona to test immediately.")

    # Persona Presets
    persona_col1, persona_col2, persona_col3, persona_col4 = st.columns(4)
    with persona_col1:
        load_custom = st.button("👤 Blank / Custom Profile", use_container_width=True)
    with persona_col2:
        load_high_risk = st.button("⚠️ Preset: High Risk Churner", use_container_width=True)
    with persona_col3:
        load_loyal = st.button("🛡️ Preset: Loyal Customer", use_container_width=True)
    with persona_col4:
        load_medium = st.button("⚖️ Preset: Moderate Risk", use_container_width=True)

    if 'customer_form' not in st.session_state:
        st.session_state.customer_form = {
            'Contract': 'Month-to-month',
            'tenure': 4,
            'MonthlyCharges': 75.50,
            'TotalCharges': 302.00,
            'PhoneService': 'Yes',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }

    if load_high_risk:
        st.session_state.customer_form = {
            'Contract': 'Month-to-month',
            'tenure': 2,
            'MonthlyCharges': 89.20,
            'TotalCharges': 178.40,
            'PhoneService': 'Yes',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }
    elif load_loyal:
        st.session_state.customer_form = {
            'Contract': 'Two year',
            'tenure': 60,
            'MonthlyCharges': 45.00,
            'TotalCharges': 2700.00,
            'PhoneService': 'Yes',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)'
        }
    elif load_medium:
        st.session_state.customer_form = {
            'Contract': 'One year',
            'tenure': 18,
            'MonthlyCharges': 65.00,
            'TotalCharges': 1170.00,
            'PhoneService': 'Yes',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Credit card (automatic)'
        }
    elif load_custom:
        st.session_state.customer_form = {
            'Contract': 'Month-to-month',
            'tenure': 12,
            'MonthlyCharges': 55.00,
            'TotalCharges': 660.00,
            'PhoneService': 'Yes',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check'
        }

    # Form layout
    with st.container():
        f1, f2, f3 = st.columns([1.1, 1.1, 1.4])
        
        with f1:
            st.markdown("##### **Account & Contract**")
            contract_opts = ['Month-to-month', 'One year', 'Two year']
            curr_c = st.session_state.customer_form['Contract']
            c_idx = contract_opts.index(curr_c) if curr_c in contract_opts else 0
            contract_input = st.selectbox("Contract Duration", contract_opts, index=c_idx)

            tenure_input = st.slider(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=int(st.session_state.customer_form['tenure']),
                help="Duration the customer has been subscribed."
            )
            
            # Show tenure bracket
            if tenure_input <= 6:
                t_bracket = "0-6 months (Critical Onboarding)"
            elif tenure_input <= 24:
                t_bracket = "7-24 months (Developing Loyalty)"
            else:
                t_bracket = "25+ months (Established Veteran)"
            st.caption(f"Tenure Cohort: **{t_bracket}**")

        with f2:
            st.markdown("##### **Charges & Billing**")
            monthly_input = st.number_input(
                "Monthly Charges ($)",
                min_value=15.0,
                max_value=150.0,
                value=float(st.session_state.customer_form['MonthlyCharges']),
                step=1.0
            )

            auto_calc_total = monthly_input * max(1, tenure_input)
            override_total = st.checkbox("Custom Total Charges", value=False)
            if override_total:
                total_input = st.number_input(
                    "Total Lifetime Charges ($)",
                    min_value=0.0,
                    value=float(st.session_state.customer_form['TotalCharges']),
                    step=10.0
                )
            else:
                total_input = auto_calc_total
                st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8; margin-top: 1.5rem;'>Estimated Lifetime Value: <b style='color:#f8fafc'>${total_input:,.2f}</b></div>", unsafe_allow_html=True)

        with f3:
            st.markdown("##### **Services & Payment Channel**")
            pm_opts = [
                'Electronic check',
                'Bank transfer (automatic)',
                'Credit card (automatic)',
                'Mailed check'
            ]
            curr_pm = st.session_state.customer_form['PaymentMethod']
            pm_idx = pm_opts.index(curr_pm) if curr_pm in pm_opts else 0
            pm_input = st.selectbox("Payment Method", pm_opts, index=pm_idx)

            s_col1, s_col2 = st.columns(2)
            with s_col1:
                phone_input = st.radio("Phone Service", ["Yes", "No"], index=0 if st.session_state.customer_form['PhoneService'] == 'Yes' else 1, horizontal=True)
            with s_col2:
                paperless_input = st.radio("Paperless Billing", ["Yes", "No"], index=0 if st.session_state.customer_form['PaperlessBilling'] == 'Yes' else 1, horizontal=True)

    # Compute Prediction
    single_df = pd.DataFrame([{
        'Contract': contract_input,
        'tenure': tenure_input,
        'MonthlyCharges': monthly_input,
        'TotalCharges': total_input,
        'PhoneService': phone_input,
        'PaperlessBilling': paperless_input,
        'PaymentMethod': pm_input
    }])

    feat_df = transform_customer_features(single_df)
    feat_scaled = models_dict['scaler'].transform(feat_df)
    churn_proba = float(active_clf.predict_proba(feat_scaled)[0, 1])
    churn_pred = int(churn_proba >= 0.5)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;' />", unsafe_allow_html=True)

    # Output Display
    res_col1, res_col2 = st.columns([1.2, 1.8])

    with res_col1:
        if churn_proba >= 0.60:
            card_class = "result-card-high"
            badge_class = "badge-red"
            risk_label = "HIGH RISK OF CHURN"
            risk_color = "#ef4444"
            icon = "🚨"
        elif churn_proba >= 0.35:
            card_class = "result-card-med"
            badge_class = "badge-amber"
            risk_label = "MODERATE / AT-RISK"
            risk_color = "#f59e0b"
            icon = "⚠️"
        else:
            card_class = "result-card-low"
            badge_class = "badge-green"
            risk_label = "HEALTHY / LOW RISK"
            risk_color = "#22c55e"
            icon = "🛡️"

        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="badge {badge_class}">{risk_label}</span>
                <span style="font-size: 1.3rem;">{icon}</span>
            </div>
            <div style="font-size: 3.2rem; font-weight: 800; color: #ffffff; margin: 0.8rem 0 0.2rem; letter-spacing: -0.03em;">
                {churn_proba * 100:.1f}%
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1;">Probability of Customer Leaving</div>
            <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; color: #94a3b8;">
                Revenue at Risk: <b style="color: #f8fafc;">${monthly_input * 12:,.2f} / year</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 24, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': risk_color, 'thickness': 0.3},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(34, 197, 94, 0.15)'},
                    {'range': [35, 60], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.15)'},
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.75,
                    'value': churn_proba * 100
                }
            }
        ))
        fig_gauge.update_layout(
            height=180,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    with res_col2:
        st.markdown("#### **Key Risk Factors & Actionable Strategy**")
        
        # Determine specific risk drivers
        drivers = []
        mitigations = []

        if contract_input == 'Month-to-month':
            drivers.append(("🔴 Month-to-Month Contract", "High exposure: Month-to-month contracts have a 42.7% historical churn rate."))
            mitigations.append("Offer a 10% discount incentive to upgrade to a 1-Year or 2-Year commitment.")
        else:
            drivers.append(("🟢 Long-Term Contract", f"Contract stability: {contract_input} contracts exhibit very low churn rates."))

        if tenure_input <= 6:
            drivers.append(("🔴 Onboarding Phase (≤ 6 mos)", "Early lifecycle drop-off: Customers in the first 6 months are at highest risk."))
            mitigations.append("Enroll customer in VIP high-touch onboarding and automated check-in calls.")
        elif tenure_input >= 25:
            drivers.append(("🟢 High Tenure (> 24 mos)", "Established customer relationship with demonstrated platform loyalty."))

        if pm_input == 'Electronic check':
            drivers.append(("🔴 Electronic Check Payment", "Highest-churn payment channel (over 45% churn rate historically)."))
            mitigations.append("Incentivize migration to Automatic Credit Card or Direct Bank Transfer ($5 billing credit).")
        else:
            drivers.append(("🟢 Reliable Payment Method", f"Using {pm_input} reduces friction and voluntary cancellation."))

        if monthly_input > 70.0:
            drivers.append(("🟡 Premium Monthly Charges", f"${monthly_input:.2f}/mo is in the top quartile of billing tiers."))
            mitigations.append("Review service usage to bundle perks or recommend an optimized cost tier.")

        # Show drivers
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("##### **Top Risk Drivers**")
            for title, desc in drivers:
                st.markdown(f"**{title}**  \n<span style='font-size:0.8rem; color:#94a3b8;'>{desc}</span>", unsafe_allow_html=True)
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        with d_col2:
            st.markdown("##### **Prescribed Retention Playbook**")
            if mitigations:
                for i, action in enumerate(mitigations, 1):
                    st.markdown(f"**{i}. {action}**")
            else:
                st.markdown("✅ **Customer is in good standing.** Maintain standard engagement cadence and monitor renewal milestones.")

        # Real-time What-If Simulator
        st.markdown("---")
        st.markdown("##### 🧪 **Interactive What-If Scenario Simulator**")
        st.caption("Simulate how proactive customer interventions would immediately reduce this customer's churn risk:")

        sim1, sim2, sim3 = st.columns(3)
        with sim1:
            sim_contract = st.selectbox("Hypothetical Contract", ["(Keep Current)", "One year", "Two year"], index=0)
        with sim2:
            sim_pm = st.selectbox("Hypothetical Payment", ["(Keep Current)", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
        with sim3:
            discount_pct = st.slider("Retention Discount", 0, 30, 0, step=5, format="%d%%")

        # Evaluate simulated customer
        sim_df = single_df.copy()
        if sim_contract != "(Keep Current)":
            sim_df['Contract'] = sim_contract
        if sim_pm != "(Keep Current)":
            sim_df['PaymentMethod'] = sim_pm
        if discount_pct > 0:
            sim_df['MonthlyCharges'] = sim_df['MonthlyCharges'] * (1 - discount_pct / 100.0)

        sim_feat = transform_customer_features(sim_df)
        sim_scaled = models_dict['scaler'].transform(sim_feat)
        sim_proba = float(active_clf.predict_proba(sim_scaled)[0, 1])

        delta_prob = (sim_proba - churn_proba) * 100
        delta_color = "#22c55e" if delta_prob < 0 else "#ef4444"
        delta_sign = "↓" if delta_prob < 0 else "↑"

        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 0.9rem 1.2rem; display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem;">
            <div>
                <span style="font-size: 0.85rem; color: #94a3b8;">Simulated Churn Probability:</span>
                <span style="font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-left: 0.5rem;">{sim_proba * 100:.1f}%</span>
            </div>
            <div>
                <span style="font-size: 0.95rem; font-weight: 700; color: {delta_color};">
                    {delta_sign} {abs(delta_prob):.1f}% Risk Reduction
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 2: Batch Customer Scoring
# =========================================================
with tab_batch:
    st.markdown("### **Batch Customer Scoring & Portfolio Risk Analysis**")
    st.caption("Upload a CSV file of customer accounts or score a batch from the repository database.")

    batch_mode = st.radio(
        "Select Data Source:",
        ["Evaluate Sample from Repository Database", "Upload Custom CSV File"],
        horizontal=True
    )

    batch_data = None

    if batch_mode == "Evaluate Sample from Repository Database":
        sample_size = st.slider("Select Sample Size to Score", 50, 1000, 200, step=50)
        batch_data = df_raw.sample(n=min(sample_size, len(df_raw)), random_state=42).copy()
    else:
        uploaded_file = st.file_uploader("Upload Customer CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(batch_data)} records!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if batch_data is not None and not batch_data.empty:
        # Check required columns
        req_cols = ['Contract', 'tenure', 'MonthlyCharges', 'TotalCharges', 'PhoneService', 'PaperlessBilling', 'PaymentMethod']
        missing = [c for c in req_cols if c not in batch_data.columns]
        
        if missing:
            st.error(f"Missing required columns in batch data: {missing}")
        else:
            with st.spinner("Executing model predictions on customer portfolio..."):
                batch_features = transform_customer_features(batch_data)
                batch_scaled = models_dict['scaler'].transform(batch_features)
                batch_probas = active_clf.predict_proba(batch_scaled)[:, 1]
                
                batch_res = batch_data.copy()
                batch_res['Churn_Probability'] = np.round(batch_probas * 100, 1)
                batch_res['Risk_Tier'] = pd.cut(
                    batch_probas,
                    bins=[-0.1, 0.35, 0.60, 1.0],
                    labels=['Low Risk', 'Moderate Risk', 'High Risk']
                )
                batch_res['Predicted_Churn'] = np.where(batch_probas >= 0.5, 'Yes', 'No')
                batch_res['Monthly_Revenue_At_Risk'] = np.where(
                    batch_res['Predicted_Churn'] == 'Yes',
                    batch_res['MonthlyCharges'],
                    0.0
                )

            # Portfolio KPIs
            total_scored = len(batch_res)
            high_risk_count = (batch_res['Risk_Tier'] == 'High Risk').sum()
            pred_churn_count = (batch_res['Predicted_Churn'] == 'Yes').sum()
            rev_at_risk = batch_res['Monthly_Revenue_At_Risk'].sum()

            bkpi1, bkpi2, bkpi3, bkpi4 = st.columns(4)
            with bkpi1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Accounts Scored</div>
                    <div class="kpi-value">{total_scored:,}</div>
                    <div class="kpi-sub">Customers in cohort</div>
                </div>
                """, unsafe_allow_html=True)
            with bkpi2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Predicted Churners</div>
                    <div class="kpi-value" style="color:#ef4444;">{pred_churn_count:,}</div>
                    <div class="kpi-sub">{(pred_churn_count/total_scored)*100:.1f}% of cohort</div>
                </div>
                """, unsafe_allow_html=True)
            with bkpi3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">High-Risk Tier</div>
                    <div class="kpi-value" style="color:#f59e0b;">{high_risk_count:,}</div>
                    <div class="kpi-sub">Probability ≥ 60%</div>
                </div>
                """, unsafe_allow_html=True)
            with bkpi4:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Monthly Rev at Risk</div>
                    <div class="kpi-value" style="color:#38bdf8;">${rev_at_risk:,.2f}</div>
                    <div class="kpi-sub">${rev_at_risk*12:,.2f} annualized</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

            # Interactive Filter & Table
            filter_col1, filter_col2 = st.columns([1, 2])
            with filter_col1:
                tier_filter = st.multiselect(
                    "Filter by Risk Tier:",
                    ['High Risk', 'Moderate Risk', 'Low Risk'],
                    default=['High Risk', 'Moderate Risk', 'Low Risk']
                )
            
            filtered_view = batch_res[batch_res['Risk_Tier'].isin(tier_filter)]

            st.dataframe(
                filtered_view[['customerID', 'Contract', 'tenure', 'MonthlyCharges', 'PaymentMethod', 'Churn_Probability', 'Risk_Tier', 'Predicted_Churn']],
                use_container_width=True,
                height=350
            )

            # Download Option
            csv_buffer = io.StringIO()
            filtered_view.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Export Scored Records as CSV",
                data=csv_buffer.getvalue(),
                file_name="churn_predictions_scored.csv",
                mime="text/csv",
                use_container_width=False
            )


# =========================================================
# TAB 3: Exploratory Data Analysis (EDA)
# =========================================================
with tab_eda:
    st.markdown("### **Exploratory Data Analysis & Business Intelligence**")
    st.caption("Statistical distribution and behavioral trends across 7,042 customer accounts in the dataset.")

    # High-level Dataset Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Customer Base</div>
            <div class="kpi-value">{len(df_raw):,}</div>
            <div class="kpi-sub">Telecom accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Historical Churn Rate</div>
            <div class="kpi-value" style="color:#ef4444;">{churn_rate:.1f}%</div>
            <div class="kpi-sub">{churn_count:,} churned customers</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        avg_tenure = df_raw['tenure'].mean()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Tenure</div>
            <div class="kpi-value">{avg_tenure:.1f} mo</div>
            <div class="kpi-sub">Across all contracts</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        churn_monthly_rev = df_raw[df_raw['Churn'] == 'Yes']['MonthlyCharges'].sum()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Lost Monthly Revenue</div>
            <div class="kpi-value" style="color:#f59e0b;">${churn_monthly_rev:,.0f}</div>
            <div class="kpi-sub">${churn_monthly_rev * 12:,.0f} / year</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Charts Grid
    c_row1_col1, c_row1_col2 = st.columns(2)
    
    with c_row1_col1:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">Customer Churn Rate by Contract Type</div>
            <div class="chart-desc">Month-to-month contracts generate over 88% of all churn occurrences.</div>
        """, unsafe_allow_html=True)
        
        contract_churn = df_raw.groupby('Contract')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='ChurnRate')
        fig_c = px.bar(
            contract_churn,
            x='Contract',
            y='ChurnRate',
            color='ChurnRate',
            color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444'],
            text=contract_churn['ChurnRate'].apply(lambda x: f"{x:.1f}%")
        )
        fig_c.update_layout(
            **PLOT_THEME,
            height=300,
            coloraxis_showscale=False,
            yaxis_title="Churn Rate (%)",
            xaxis_title=""
        )
        fig_c.update_traces(textposition='outside')
        st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c_row1_col2:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">Churn Rate by Payment Method</div>
            <div class="chart-desc">Customers paying via Electronic Check have significantly higher churn than automated methods.</div>
        """, unsafe_allow_html=True)
        
        pm_churn = df_raw.groupby('PaymentMethod')['Churn'].apply(lambda s: (s == 'Yes').mean() * 100).reset_index(name='ChurnRate')
        pm_churn = pm_churn.sort_values(by='ChurnRate', ascending=True)
        fig_pm = px.bar(
            pm_churn,
            y='PaymentMethod',
            x='ChurnRate',
            orientation='h',
            color='ChurnRate',
            color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444'],
            text=pm_churn['ChurnRate'].apply(lambda x: f"{x:.1f}%")
        )
        fig_pm.update_layout(
            **PLOT_THEME,
            height=300,
            coloraxis_showscale=False,
            xaxis_title="Churn Rate (%)",
            yaxis_title=""
        )
        fig_pm.update_traces(textposition='outside')
        st.plotly_chart(fig_pm, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    c_row2_col1, c_row2_col2 = st.columns(2)

    with c_row2_col1:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">Customer Tenure Distribution vs Churn Status</div>
            <div class="chart-desc">Newer customers churn in high numbers, stabilizing past month 20.</div>
        """, unsafe_allow_html=True)
        
        fig_tenure = px.histogram(
            df_raw,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=36,
            color_discrete_map={"No": "#3b82f6", "Yes": "#ef4444"},
            opacity=0.75
        )
        fig_tenure.update_layout(
            **PLOT_THEME,
            height=300,
            xaxis_title="Tenure (Months)",
            yaxis_title="Customer Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tenure, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c_row2_col2:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">Monthly Charges Distribution</div>
            <div class="chart-desc">Higher monthly charges ($70 - $100) are strongly correlated with elevated churn rates.</div>
        """, unsafe_allow_html=True)

        fig_charges = px.box(
            df_raw,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"No": "#22c55e", "Yes": "#ef4444"},
            points=False
        )
        fig_charges.update_layout(
            **PLOT_THEME,
            height=300,
            xaxis_title="Customer Churn Status",
            yaxis_title="Monthly Charges ($)",
            showlegend=False
        )
        st.plotly_chart(fig_charges, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 4: Model Benchmarks & Explainability
# =========================================================
with tab_benchmarks:
    st.markdown("### **Machine Learning Model Evaluation & Benchmark Results**")
    st.caption("Comparison of 8 supervised algorithms evaluated using 5-Fold Cross-Validation and a 20% holdout test set.")

    if df_benchmarks is not None:
        st.dataframe(
            df_benchmarks.style.highlight_max(subset=['Test ROC-AUC', 'Test Accuracy', 'Test Recall', 'Test F1-Score'], color='#1e3a8a'),
            use_container_width=True
        )

    b_col1, b_col2 = st.columns(2)

    with b_col1:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">Global Feature Importances (Random Forest)</div>
            <div class="chart-desc">Contract type and billing variables dominate model decision-making.</div>
        """, unsafe_allow_html=True)

        rf = models_dict['rf_model']
        fi_series = pd.Series(rf.feature_importances_, index=models_dict['feature_names']).sort_values(ascending=True)
        
        # Friendly feature names
        friendly_names = {
            'ord__Contract': 'Contract Duration',
            'remainder__MonthlyCharges': 'Monthly Charges ($)',
            'remainder__tenure': 'Tenure (Months)',
            'remainder__TotalCharges': 'Total Charges ($)',
            'nom__PaymentMethod_Electronic check': 'Payment: Electronic Check',
            'ord__Tenure_Group': 'Tenure Cohort (0-6, 7-24, 25+)',
            'nom__PaperlessBilling_Yes': 'Paperless Billing: Yes',
            'nom__PaymentMethod_Mailed check': 'Payment: Mailed Check',
            'nom__PhoneService_Yes': 'Phone Service: Yes',
            'nom__PaymentMethod_Credit card (automatic)': 'Payment: Credit Card (Auto)'
        }
        fi_df = pd.DataFrame({
            'Feature': [friendly_names.get(k, k) for k in fi_series.index],
            'Importance': fi_series.values
        })

        fig_fi = px.bar(
            fi_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_fi.update_layout(
            **PLOT_THEME,
            height=320,
            coloraxis_showscale=False,
            xaxis_title="Relative Feature Importance",
            yaxis_title=""
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with b_col2:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">Confusion Matrix on Test Set (1,409 Samples)</div>
            <div class="chart-desc">Evaluation of true negatives, false positives, false negatives, and true positives.</div>
        """, unsafe_allow_html=True)

        from sklearn.metrics import confusion_matrix
        y_test = models_dict['y_test']
        x_test_scaled = models_dict['x_test_scaled']
        y_pred = active_clf.predict(x_test_scaled)
        cm = confusion_matrix(y_test, y_pred)

        cm_labels = [["True Negatives<br>(Stayed)", "False Positives<br>(False Alarm)"],
                     ["False Negatives<br>(Missed Churn)", "True Positives<br>(Caught Churn)"]]

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            text=[[f"{cm[i][j]}<br><span style='font-size:10px'>{cm_labels[i][j]}</span>" for j in range(2)] for i in range(2)],
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ))
        fig_cm.update_layout(
            **PLOT_THEME,
            height=320,
            xaxis_title="Predicted Class",
            yaxis_title="Ground Truth"
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 5: Architecture & Documentation
# =========================================================
with tab_about:
    st.markdown("### **System Architecture & Technical Documentation**")
    st.markdown("""
    #### 🛡️ **Project Background**
    **Barrier-** is a customer churn prediction and retention platform developed as an NTI Graduation project. 
    Customer attrition represents a major loss of monthly recurring revenue for service providers. By applying machine learning to behavioral signals, **Barrier-** identifies at-risk accounts early and recommends personalized intervention strategies.

    ---

    #### ⚙️ **End-to-End Pipeline Architecture**
    1. **Data Ingestion**:
       - 7,042 customer records loaded from `data/churn_data.csv`.
       - Data cleaning: coerced missing spaces in `TotalCharges` to `0.0`.
    2. **Feature Engineering**:
       - `Tenure_Group` ordinal binning: `0-6 months`, `7-24 months`, `25+ months`.
       - Ordinal Encoding for Contract duration (`Month-to-month` → 0, `One year` → 1, `Two year` → 2).
       - One-Hot Encoding for nominal categorical features (`PhoneService`, `PaperlessBilling`, `PaymentMethod`).
       - Feature scaling with `StandardScaler`.
    3. **Model Selection & Tuning**:
       - 8 algorithms evaluated via 5-Fold Stratified Cross-Validation (`GridSearchCV`).
       - Optimized for **ROC-AUC** to balance precision and recall on imbalanced class distributions.
       - **Top Performer**: Tuned Random Forest (`n_estimators=200, max_depth=8, class_weight='balanced'`) achieving **0.8446 Test ROC-AUC** and **75.9% Recall**.
    4. **Inference & UI Delivery**:
       - Real-time interactive Streamlit web application.
       - What-If scenario retention simulation.
       - Batch customer scoring and CSV export.

    ---

    #### 💻 **Technology Stack**
    - **Language**: Python 3.12+
    - **Frontend**: Streamlit, Plotly Express & Graph Objects, Custom CSS
    - **Machine Learning**: Scikit-Learn, XGBoost, Joblib
    - **Data Manipulation**: Pandas, NumPy
    - **Repository**: [GitHub: Mostafa-Ashraf-Elshahawy/Barrier-](https://github.com/Mostafa-Ashraf-Elshahawy/Barrier-)
    """)
