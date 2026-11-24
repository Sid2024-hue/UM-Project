import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import streamlit as st

st.set_page_config(page_title="European Bank Churn Dashboard", layout="wide")
st.title("European Bank Customer Churn Dashboard")

FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary']


# ---------- Data ----------
@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    return df.drop(columns=['Year', 'CustomerId', 'Surname'], errors='ignore')


st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload European_Bank.csv", type="csv")
try:
    df_all = load_data(uploaded if uploaded else "European_Bank.csv")
except FileNotFoundError:
    st.warning("Upload European_Bank.csv in the sidebar (or place it next to app.py).")
    st.stop()

# ---------- Filters ----------
st.sidebar.header("Filters")
geos = st.sidebar.multiselect("Geography", sorted(df_all['Geography'].unique()),
                              default=sorted(df_all['Geography'].unique()))
genders = st.sidebar.multiselect("Gender", sorted(df_all['Gender'].unique()),
                                 default=sorted(df_all['Gender'].unique()))
age_min, age_max = int(df_all['Age'].min()), int(df_all['Age'].max())
age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
products = st.sidebar.multiselect("Number of products", sorted(df_all['NumOfProducts'].unique()),
                                  default=sorted(df_all['NumOfProducts'].unique()))

df = df_all[
    df_all['Geography'].isin(geos)
    & df_all['Gender'].isin(genders)
    & df_all['Age'].between(*age_range)
    & df_all['NumOfProducts'].isin(products)
]

if df.empty:
    st.error("No rows match the current filters.")
    st.stop()

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Churn rate", f"{df['Exited'].mean():.1%}")
c3.metric("Avg. age", f"{df['Age'].mean():.1f}")
c4.metric("Avg. balance", f"{df['Balance'].mean():,.0f}")

tab_data, tab_eda, tab_rel, tab_model, tab_pred = st.tabs(
    ["Data", "Churn segments", "Relationships", "Logit model", "Predict"])

# ---------- Data ----------
with tab_data:
    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("One-hot encoded (Geography, Gender)")
    df_encoded = pd.get_dummies(df, columns=['Geography', 'Gender'], dtype=int)
    st.dataframe(df_encoded.head(20), use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# ---------- Segments ----------
with tab_eda:
    st.subheader("Churn rate by segment")
    seg = st.selectbox("Segment by", ['Geography', 'Gender', 'NumOfProducts',
                                      'HasCrCard', 'IsActiveMember', 'Tenure'])
    rate = df.groupby(seg)['Exited'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=rate, x=seg, y='Exited', ax=ax)
    ax.set_ylabel("Churn rate")
    st.pyplot(fig)

    st.subheader("Distribution by churn status")
    num = st.selectbox("Variable", ['Age', 'Balance', 'CreditScore', 'EstimatedSalary'])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x=num, hue='Exited', kde=True, element='step', ax=ax)
    st.pyplot(fig)

# ---------- Relationships ----------
with tab_rel:
    st.subheader("Correlation heatmap")
    cor = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cor, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Variable vs Exited")
    var = st.selectbox("X variable", ['Age', 'Balance', 'CreditScore',
                                      'Tenure', 'EstimatedSalary'])
    left, right = st.columns(2)
    with left:
        st.caption("Scatter")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df, x=var, y='Exited', ax=ax, alpha=0.4)
        st.pyplot(fig)
    with right:
        st.caption("Regression fit")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.regplot(data=df, x=var, y='Exited', ax=ax,
                    scatter_kws={'alpha': 0.2})
        st.pyplot(fig)


# ---------- Model ----------
@st.cache_resource
def fit_logit(data):
    X = sm.add_constant(data[FEATURES])
    return sm.Logit(data['Exited'], X).fit(disp=0)


with tab_model:
    st.subheader("Logistic regression (fit on filtered data)")
    if df['Exited'].nunique() < 2:
        st.warning("Filtered data has only one outcome class; cannot fit the model.")
        result = None
    else:
        try:
            result = fit_logit(df)
        except Exception as e:  # singular matrix etc.
            st.error(f"Model failed to fit: {e}")
            result = None

    if result is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Pseudo R²", f"{result.prsquared:.4f}")
        m2.metric("Log-likelihood", f"{result.llf:,.1f}")
        m3.metric("Observations", f"{int(result.nobs):,}")

        ci = result.conf_int()
        coefs = pd.DataFrame({
            'coef': result.params,
            'std err': result.bse,
            'z': result.tvalues,
            'P>|z|': result.pvalues,
            'odds ratio': np.exp(result.params),
            'CI low': ci[0],
            'CI high': ci[1],
        })
        st.dataframe(coefs.style.format("{:.4g}"), use_container_width=True)

        sig = coefs.drop(index='const')
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['tab:red' if p < 0.05 else 'gray' for p in sig['P>|z|']]
        ax.barh(sig.index, sig['coef'], color=colors)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title("Coefficients (red = p < 0.05; unscaled features)")
        st.pyplot(fig)

        with st.expander("Full statsmodels summary"):
            st.text(result.summary().as_text())

# ---------- Predict ----------
with tab_pred:
    st.subheader("Predict churn probability for a customer")
    if result is None:
        st.info("Model unavailable for the current filters.")
    else:
        a, b = st.columns(2)
        with a:
            cs = st.number_input("CreditScore", 300, 900, 650)
            age = st.number_input("Age", 18, 100, 40)
            tenure = st.number_input("Tenure", 0, 10, 5)
            balance = st.number_input("Balance", 0.0, 300000.0, 75000.0, step=1000.0)
        with b:
            nprod = st.number_input("NumOfProducts", 1, 4, 1)
            card = st.selectbox("HasCrCard", [1, 0])
            active = st.selectbox("IsActiveMember", [1, 0])
            salary = st.number_input("EstimatedSalary", 0.0, 250000.0, 100000.0, step=1000.0)

        row = pd.DataFrame([[1.0, cs, age, tenure, balance, nprod, card, active, salary]],
                           columns=['const'] + FEATURES)
        prob = float(result.predict(row)[0])
        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
    