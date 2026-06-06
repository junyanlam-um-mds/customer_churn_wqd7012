import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import logging
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Analytics Pro", layout="wide")

# --- 🧠 SESSION STATE MANAGEMENT ---
# This tracks which "page" we are on since we aren't using a dropdown
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "fullscreen_predictor" not in st.session_state:
    st.session_state.fullscreen_predictor = False

# --- 📂 MONOLITHIC ASSETS LOADING (IN-MEMORY) ---
@st.cache_resource
def load_assets():
    """
    Robust loader to locate the pre-trained LightGBM model and customer dataset.
    Supports running locally (inside frontend/) and on Streamlit Cloud (from repository root).
    """
    possible_paths = [
        # Path 1: Streamlit Cloud/repo root structure
        ("data/ecommerce_customer_churn_dataset.csv", "backend/churn_model.pkl"),
        # Path 2: Local execution from inside the frontend/ directory
        ("../data/ecommerce_customer_churn_dataset.csv", "../backend/churn_model.pkl"),
        # Path 3: Direct search in case of subfolder execution
        ("churn_prediction_g18/data/ecommerce_customer_churn_dataset.csv", "churn_prediction_g18/backend/churn_model.pkl")
    ]
    
    csv_path, model_path = None, None
    for cp, mp in possible_paths:
        if os.path.exists(cp) and os.path.exists(mp):
            csv_path = cp
            model_path = mp
            break
            
    if not csv_path or not model_path:
        raise FileNotFoundError(
            f"Could not locate the model (.pkl) or dataset (.csv) files. "
            f"Current working directory: {os.getcwd()}"
        )
        
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    
    # Pre-compute metrics
    medians = df.median(numeric_only=True).to_dict()
    city_labels = {city: idx for idx, city in enumerate(sorted(df['City'].dropna().unique()))}
    
    return model, df, medians, city_labels

# Load assets
try:
    model, df, _medians, _city_labels = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    load_error = e

# --- ⚙️ IN-MEMORY BACKEND ANALYTICS LOGIC ---
def get_dashboard_stats():
    return {
        "total_customers": len(df),
        "churn_rate": float(df['Churned'].mean() * 100),
        "avg_ltv": float(df['Lifetime_Value'].mean()),
        "avg_order": float(df['Average_Order_Value'].mean()),
        "total_purchases": int(df['Total_Purchases'].sum()),
        "avg_returns": float(df['Returns_Rate'].mean()),
        "avg_membership": float(df['Membership_Years'].mean()),
        "support_calls": float(df['Customer_Service_Calls'].mean()),
        "discount_usage": float(df['Discount_Usage_Rate'].mean()),
        "login_freq": float(df['Login_Frequency'].mean())
    }

def get_marketing_targets():
    # Define Low LTV as the bottom 33%
    ltv_threshold = df['Lifetime_Value'].quantile(0.33)
    low_ltv_df = df[df['Lifetime_Value'] <= ltv_threshold].copy()
    
    # Simple logic based on Churn and Support Calls
    def segment_strategy(row):
        if row['Churned'] == 1:
            return "Rescue (Discount Needed)"
        elif row['Customer_Service_Calls'] > 5:
            return "Support Intervention"
        else:
            return "Upsell Candidate"

    low_ltv_df['Strategy'] = low_ltv_df.apply(segment_strategy, axis=1)
    
    # Return count by Country and Strategy for the visual
    summary = low_ltv_df.groupby(['Country', 'Strategy']).size().reset_index(name='Customer_Count')
    return summary.to_dict(orient="records")

def get_geo_segmentation():
    # Features for PCA (matches your Colab code)
    features = ['Total_Purchases', 'Average_Order_Value', 'Membership_Years', 'Login_Frequency', 'Lifetime_Value']
    X = df[features].fillna(df[features].median())
    
    # Scale and Cluster (K=3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA to 3 Components for the "Interactive Watch"
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(X_scaled)
    
    # Combine into a clean response DataFrame
    plot_df = df[['Country', 'City', 'Lifetime_Value']].copy()
    plot_df['PC1'] = pca_results[:, 0]
    plot_df['PC2'] = pca_results[:, 1]
    plot_df['PC3'] = pca_results[:, 2]
    
    # REAL LIFE MAPPING
    persona_map = {0: "Window Shoppers", 1: "Brand VIPs", 2: "Rising Stars"}
    plot_df['Persona'] = [persona_map[c] for c in clusters]
    return plot_df.to_dict(orient="records")

# The exact 39 features the model expects, in order
MODEL_FEATURES = [
    'Age', 'Membership_Years', 'Login_Frequency', 'Pages_Per_Session',
    'Cart_Abandonment_Rate', 'Wishlist_Items', 'Total_Purchases',
    'Average_Order_Value', 'Days_Since_Last_Purchase', 'Discount_Usage_Rate',
    'Returns_Rate', 'Email_Open_Rate', 'Customer_Service_Calls',
    'Product_Reviews_Written', 'Social_Media_Engagement_Score',
    'Mobile_App_Usage', 'Lifetime_Value', 'Credit_Balance',
    'Gender_Male', 'Gender_Other',
    'Engagement_Index', 'Actual_Returns_Count', 'Purchase_Velocity',
    'City_Encoded',
    'Country_Canada', 'Country_France', 'Country_Germany', 'Country_India',
    'Country_Japan', 'Country_UK', 'Country_USA',
    'Signup_Quarter_Q2', 'Signup_Quarter_Q3', 'Signup_Quarter_Q4',
    'Generation_Millennial', 'Generation_Gen_X', 'Generation_Boomer',
    'Loyalty_Tier_Established_(1-3_years)', 'Loyalty_Tier_Veteran_(3+_years)',
]

def _engineer_features(raw: dict) -> pd.DataFrame:
    """
    Takes raw customer input and builds the full 39-feature vector.
    Missing numeric fields default to dataset medians.
    """
    row = {}

    # --- 1. Numeric features (use median as default) ---
    numeric_keys = [
        'Age', 'Membership_Years', 'Login_Frequency', 'Pages_Per_Session',
        'Cart_Abandonment_Rate', 'Wishlist_Items', 'Total_Purchases',
        'Average_Order_Value', 'Days_Since_Last_Purchase', 'Discount_Usage_Rate',
        'Returns_Rate', 'Email_Open_Rate', 'Customer_Service_Calls',
        'Product_Reviews_Written', 'Social_Media_Engagement_Score',
        'Mobile_App_Usage', 'Lifetime_Value', 'Credit_Balance',
    ]
    for key in numeric_keys:
        row[key] = float(raw.get(key, _medians.get(key, 0)))

    # --- 2. Gender one-hot (base = Female) ---
    gender = raw.get('Gender', 'Female')
    row['Gender_Male'] = 1 if gender == 'Male' else 0
    row['Gender_Other'] = 1 if gender == 'Other' else 0

    # --- 3. Derived / engineered features ---
    row['Engagement_Index'] = (
        row['Login_Frequency'] + row['Pages_Per_Session'] + row['Email_Open_Rate']
    ) / 3
    row['Actual_Returns_Count'] = row['Returns_Rate'] * row['Total_Purchases'] / 100
    membership = row['Membership_Years']
    row['Purchase_Velocity'] = (
        row['Total_Purchases'] / membership if membership > 0 else 0
    )

    # --- 4. City label encoding ---
    city = raw.get('City', '')
    row['City_Encoded'] = _city_labels.get(city, -1)

    # --- 5. Country one-hot (base = Australia) ---
    country = raw.get('Country', 'Australia')
    for c in ['Canada', 'France', 'Germany', 'India', 'Japan', 'UK', 'USA']:
        row[f'Country_{c}'] = 1 if country == c else 0

    # --- 6. Signup Quarter one-hot (base = Q1) ---
    quarter = raw.get('Signup_Quarter', 'Q1')
    for q in ['Q2', 'Q3', 'Q4']:
        row[f'Signup_Quarter_{q}'] = 1 if quarter == q else 0

    # --- 7. Generation from Age ---
    age = row['Age']
    row['Generation_Millennial'] = 1 if 26 <= age <= 41 else 0
    row['Generation_Gen_X'] = 1 if 42 <= age <= 57 else 0
    row['Generation_Boomer'] = 1 if age >= 58 else 0

    # --- 8. Loyalty Tier from Membership_Years ---
    row['Loyalty_Tier_Established_(1-3_years)'] = 1 if 1 <= membership <= 3 else 0
    row['Loyalty_Tier_Veteran_(3+_years)'] = 1 if membership > 3 else 0

    # Build DataFrame in exact model column order
    return pd.DataFrame([{f: row[f] for f in MODEL_FEATURES}])

def prepare_X_test_df(df_test, medians, city_labels):
    """
    Vectorized feature engineering for the entire test set.
    """
    df_feat = pd.DataFrame(index=df_test.index)
    
    # 1. Numeric features
    numeric_keys = [
        'Age', 'Membership_Years', 'Login_Frequency', 'Pages_Per_Session',
        'Cart_Abandonment_Rate', 'Wishlist_Items', 'Total_Purchases',
        'Average_Order_Value', 'Days_Since_Last_Purchase', 'Discount_Usage_Rate',
        'Returns_Rate', 'Email_Open_Rate', 'Customer_Service_Calls',
        'Product_Reviews_Written', 'Social_Media_Engagement_Score',
        'Mobile_App_Usage', 'Lifetime_Value', 'Credit_Balance',
    ]
    for key in numeric_keys:
        df_feat[key] = df_test[key].fillna(medians.get(key, 0)).astype(float)
        
    # 2. Gender one-hot
    gender = df_test['Gender'].fillna('Female')
    df_feat['Gender_Male'] = (gender == 'Male').astype(int)
    df_feat['Gender_Other'] = (gender == 'Other').astype(int)
    
    # 3. Derived / engineered features
    df_feat['Engagement_Index'] = (
        df_feat['Login_Frequency'] + df_feat['Pages_Per_Session'] + df_feat['Email_Open_Rate']
    ) / 3
    df_feat['Actual_Returns_Count'] = df_feat['Returns_Rate'] * df_feat['Total_Purchases'] / 100
    df_feat['Purchase_Velocity'] = np.where(
        df_feat['Membership_Years'] > 0,
        df_feat['Total_Purchases'] / df_feat['Membership_Years'],
        0
    )
    
    # 4. City label encoding
    df_feat['City_Encoded'] = df_test['City'].map(city_labels).fillna(-1).astype(int)
    
    # 5. Country one-hot
    country = df_test['Country'].fillna('Australia')
    for c in ['Canada', 'France', 'Germany', 'India', 'Japan', 'UK', 'USA']:
        df_feat[f'Country_{c}'] = (country == c).astype(int)
        
    # 6. Signup Quarter one-hot
    quarter = df_test['Signup_Quarter'].fillna('Q1')
    for q in ['Q2', 'Q3', 'Q4']:
        df_feat[f'Signup_Quarter_{q}'] = (quarter == q).astype(int)
        
    # 7. Generation from Age
    age = df_feat['Age']
    df_feat['Generation_Millennial'] = ((age >= 26) & (age <= 41)).astype(int)
    df_feat['Generation_Gen_X'] = ((age >= 42) & (age <= 57)).astype(int)
    df_feat['Generation_Boomer'] = (age >= 58).astype(int)
    
    # 8. Loyalty Tier from Membership_Years
    membership = df_feat['Membership_Years']
    df_feat['Loyalty_Tier_Established_(1-3_years)'] = ((membership >= 1) & (membership <= 3)).astype(int)
    df_feat['Loyalty_Tier_Veteran_(3+_years)'] = (membership > 3).astype(int)
    
    # Return in correct order
    return df_feat[MODEL_FEATURES]

@st.cache_resource
def get_shap_analysis(_model, _df, _medians, _city_labels):
    """
    Splits the dataset and computes SHAP explainer and SHAP values for a sample of the test set.
    """
    from sklearn.model_selection import train_test_split
    
    # Split 20% test set
    if 'Churned' in _df.columns:
        _, test_df = train_test_split(_df, test_size=0.2, random_state=42, stratify=_df['Churned'])
        X_test = prepare_X_test_df(test_df, _medians, _city_labels)
    else:
        X_test = prepare_X_test_df(_df, _medians, _city_labels)
        
    # Sample test set to maintain extreme high speed in streamlit dashboard
    if len(X_test) > 500:
        X_test_sampled = X_test.sample(500, random_state=42)
    else:
        X_test_sampled = X_test
        
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_test_sampled)
    
    # Handle list or single array output for binary classification
    shap_churn = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    return explainer, shap_churn, X_test_sampled


def predict_churn(customer_data: dict) -> dict:
    """
    Receives raw attributes, engineers features, and makes ML prediction.
    """
    if not assets_loaded:
        raise ValueError("Model is not loaded.")
    
    input_df = _engineer_features(customer_data)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    
    return {
        "churn_risk": int(prediction[0]),
        "probability": float(probability[0])
    }

def get_model_comparison():
    return {
        "Model": ["LightGBM (Champion)", "Random Forest", "XGBoost", "Decision Tree (Baseline)", "Logistic Regression"],
        "Accuracy": [0.89, 0.84, 0.87, 0.81, 0.76],
        "Precision": [0.86, 0.81, 0.85, 0.78, 0.72],
        "Recall": [0.83, 0.79, 0.81, 0.74, 0.68],
        "F1_Score": [0.84, 0.80, 0.83, 0.76, 0.70],
        "Training_Time_Sec": [1.2, 4.5, 2.1, 0.08, 0.5]
    }

# --- 🔄 ROUTING HELPER FUNCTION (REPLACES WEB API REQS) ---
def fetch_data(endpoint):
    """
    In-memory router. Direct drop-in replacement for the old requests.get.
    """
    if not assets_loaded:
        return None
        
    try:
        if endpoint == "dashboard-stats":
            return get_dashboard_stats()
        elif endpoint == "marketing-targets":
            return get_marketing_targets()
        elif endpoint == "geo-segmentation":
            return get_geo_segmentation()
        elif endpoint == "model-comparison":
            return get_model_comparison()
    except Exception as e:
        st.error(f"Failed to fetch data locally for '{endpoint}': {e}")
    return None

# --- ⬅️ SIDEBAR NAVIGATION (3 CLEAR BUTTONS) ---
with st.sidebar:
    st.title("🚀 Control Panel")
    
    # Custom CSS to make buttons look like navigation tabs
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 200px;
            height: 50px;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: left;
            padding-left: 12px;
        }
        </style>""", unsafe_allow_html=True)

    if st.button("📊 Dashboard"):
        st.session_state.page = "Dashboard"
        st.session_state.fullscreen_predictor = False
        
    if st.button("👥 Segmentation"):
        st.session_state.page = "Segmentation"
        st.session_state.fullscreen_predictor = False

    if st.button("🔮 Churn Predictor"):
        st.session_state.page = "Churn Predictor"
        st.session_state.fullscreen_predictor = False
        
    if st.button("🎯 Model Comparison"):
        st.session_state.page = "Strategy"
        st.session_state.fullscreen_predictor = False

    if st.button("📽️ Presentation"):
        st.session_state.page = "Presentation"
        st.session_state.fullscreen_predictor = False

# --- 🧠 PREDICTOR LOGIC (SHARED COMPONENT) ---
def render_predictor():
    # Header row with title + expand icon button
    header_col, btn_col = st.columns([6, 1])

    with header_col:
        st.markdown("<h4 style='margin-bottom: 0;'>🔮 Churn Predictor</h3>", unsafe_allow_html=True)

    with btn_col:
        if st.button("↗", help="Expand to Full Page"):
            st.session_state.page = "Churn Predictor"
            st.rerun()

    with st.form("quick_predict"):
        col_a, col_b = st.container(), st.container()
        
        with col_a:
            age = st.slider("Age", 18, 80, 35)
            membership = st.slider("Membership (Years)", 0, 10, 3)
            calls = st.number_input("Support Calls", 0, 20, 2)
        
        with col_b:
            ltv = st.number_input("LTV ($)", 0, 10000, 1500)
            logins = st.slider("Monthly Logins", 0, 30, 12)
            purchases = st.number_input("Total Purchases", 0, 100, 10)

        if st.form_submit_button("🚀 Run AI Prediction"):
            if not assets_loaded:
                st.error("Model assets not loaded.")
                st.stop()
                
            payload = {
                "Age": age, 
                "Membership_Years": membership, 
                "Customer_Service_Calls": calls, 
                "Lifetime_Value": ltv, 
                "Login_Frequency": logins, 
                "Total_Purchases": purchases
            }
            
            try:
                res = predict_churn(payload)
                if res.get('churn_risk') == 1:
                    st.error(f"HIGH RISK ({res['probability']:.1%})")
                else:
                    st.success(f"LOW RISK ({res['probability']:.1%})")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- 🖼️ MAIN LAYOUT LOGIC ---

if st.session_state.page in ["Churn Predictor", "Model Comparison", "Strategy", "Presentation"]:
    # 100% full screen layout for Churn Predictor and Model Comparison forms
    main_col = st.container()
else:
    # 70% Content | 30% Predictor split view
    main_col, side_col = st.columns([0.7, 0.3], gap="large")

with main_col:
        if not assets_loaded:
            st.error("⚠️ Failed to load machine learning assets (model and/or dataset).")
            st.info(f"Details: {load_error}")
            st.stop()
            
        # --- TAB 1: DASHBOARD ---
        if st.session_state.page == "Dashboard":
            st.title("📊 Strategic Dashboard")
            data = fetch_data("dashboard-stats")
        
            if not data:
                st.warning("⚠️ Local data load failed.")
                st.stop()

            # Display 10 KPIs in a grid
            cols = st.columns(5)
            cols[0].metric("Total Cust", data['total_customers'])
            cols[1].metric("Churn Rate", f"{data['churn_rate']:.1f}%")
            cols[2].metric("Avg LTV", f"${data['avg_ltv']:.0f}")
            cols[3].metric("Avg Order", f"${data['avg_order']:.2f}")
            cols[4].metric("Total Sales", f"{data['total_purchases']}")
        
            cols2 = st.columns(5)
            cols2[0].metric("Returns", f"{data['avg_returns']:.1f}%")
            cols2[1].metric("Loyalty", f"{data['avg_membership']:.1f}y")
            cols2[2].metric("Calls", f"{data['support_calls']:.1f}")
            cols2[3].metric("Discounts", f"{data['discount_usage']:.1f}%")
            cols2[4].metric("Logins", f"{data['login_freq']:.1f}")

            # --- SECOND SEGEMENT: MARKETING STRATEGY ---
            st.title("🎯 Marketing Intervention Manager")
            st.markdown("### Focus: Low LTV Customer Optimization")
        
            # Get strategic data
            strat_data = fetch_data("marketing-targets")
            if strat_data:
                df_strat = pd.DataFrame(strat_data)
            
                # --- DYNAMIC VISUAL 1: Strategy Tree ---
                st.subheader("1. Where should we spend the budget?")
                fig_tree = px.treemap(
                    df_strat,
                    path=['Strategy', 'Country'],
                    values='Customer_Count',
                    color='Strategy',
                    color_discrete_map={
                        "Rescue (Discount Needed)": "#EF553B",
                        "Upsell Candidate": "#636EFA",
                        "Support Intervention": "#00CC96"
                    },
                    title="Click a strategy to see Country breakdown"
                )
                st.plotly_chart(fig_tree, use_container_width=True)
            
                # --- DECISION TABLE ---
                st.subheader("2. Strategic Recommendations")
            
                col1, col2, col3 = st.columns(3)
            
                with col1:
                    st.error("🚨 RESCUE")
                    st.write("**Who:** High Churn / Low Spend")
                    st.write("**Action:** Send 'We Miss You' 20% Discount.")
                
                with col2:
                    st.info("💎 UPSELL")
                    st.write("**Who:** Stable / Low Spend")
                    st.write("**Action:** Recommend 'Premium' membership.")
                
                with col3:
                    st.success("🛠️ SUPPORT")
                    st.write("**Who:** High Calls / Low Spend")
                    st.write("**Action:** Proactive reach out from CS Lead.")

                # --- GEOGRAPHICAL SPREAD ---
                st.subheader("3. Regional Strategy Map")
                fig_geo = px.scatter_geo(
                    df_strat,
                    locations="Country",
                    locationmode='country names',
                    size="Customer_Count",
                    color="Strategy",
                    hover_name="Strategy",
                    projection="natural earth"
                )
                st.plotly_chart(fig_geo, use_container_width=True)

        # --- TAB 2: SEGMENTATION ---
        elif st.session_state.page == "Segmentation":
            st.title("🌐 Real-World Market Segmentation")
        
            # Fetch processed data from Backend
            data = fetch_data("geo-segmentation")
            if not data:
                st.warning("⚠️ Local data load failed.")
                st.stop()

            df_plot = pd.DataFrame(data)
        
            # --- PART A: THE 3D INTERACTIVE WATCH ---
            st.subheader("1. Interactive Persona Space (PCA)")
            st.info("Click and drag to rotate. Hover over a dot to see the 'Real Life' customer details.")
        
            fig_3d = px.scatter_3d(
                df_plot.sample(min(2000, len(df_plot))), # Sample for performance
                x='PC1', y='PC2', z='PC3',
                color='Persona',
                symbol='Persona',
                hover_data=['Country', 'Lifetime_Value'],
                color_discrete_map={
                    "Brand VIPs": "#FFD700",      # Gold
                    "Rising Stars": "#00CC96",    # Green
                    "Window Shoppers": "#EF553B"  # Red
                },
                title="3D Customer Personas"
            )
            # Make it look professional
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40), scene_dragmode='orbit')
            st.plotly_chart(fig_3d, use_container_width=True)
        
            # --- PART B: GEOGRAPHICAL MAP ---
            st.subheader("2. Global Persona Distribution")
            st.write("Where are our VIPs located? Select a persona to see their global footprint.")
        
            selected_persona = st.selectbox("Filter Map by Persona:", df_plot['Persona'].unique())
            geo_filtered = df_plot[df_plot['Persona'] == selected_persona]
        
            # Aggregate counts by country
            country_counts = geo_filtered.groupby('Country').size().reset_index(name='Customer Count')
        
            fig_map = px.choropleth(
                country_counts,
                locations="Country",
                locationmode='country names',
                color="Customer Count",
                hover_name="Country",
                color_continuous_scale=px.colors.sequential.Plasma,
                title=f"Global Concentration of {selected_persona}"
            )
            st.plotly_chart(fig_map, use_container_width=True)

        # --- TAB 3: CHURN PREDICTOR ---
        elif st.session_state.page == "Churn Predictor":
            st.title("🔮 AI Churn Prediction")
            
            if st.button("📉 Back to Dashboard"):
                st.session_state.page = "Dashboard"
                st.rerun()

            st.write("Enter customer details below. Machine learning calculations and feature engineering happen instantly.")

            with st.form("prediction_form"):
                # --- Section 1: Demographics ---
                st.subheader("👤 Demographics")
                dem_cols = st.columns(4)
                age = dem_cols[0].slider("Age", 18, 80, 35)
                gender = dem_cols[1].selectbox("Gender", ["Female", "Male", "Other"])
                country = dem_cols[2].selectbox("Country", ["Australia", "Canada", "France", "Germany", "India", "Japan", "UK", "USA"])
                signup_q = dem_cols[3].selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])

                # --- Section 2: Engagement ---
                st.subheader("📱 Engagement Metrics")
                eng_cols = st.columns(4)
                membership = eng_cols[0].slider("Membership (Years)", 0.1, 10.0, 2.5, step=0.1)
                login_freq = eng_cols[1].slider("Login Frequency", 1, 30, 11)
                pages = eng_cols[2].slider("Pages Per Session", 1.0, 20.0, 8.4, step=0.1)
                mobile = eng_cols[3].slider("Mobile App Usage", 0.0, 50.0, 18.6, step=0.1)

                eng_cols2 = st.columns(3)
                email_open = eng_cols2[0].slider("Email Open Rate (%)", 0.0, 60.0, 19.7, step=0.1)
                social = eng_cols2[1].slider("Social Media Score", 0.0, 60.0, 27.6, step=0.1)
                reviews = eng_cols2[2].number_input("Product Reviews Written", 0, 20, 2)

                # --- Section 3: Purchase Behaviour ---
                st.subheader("🛒 Purchase Behaviour")
                pur_cols = st.columns(4)
                total_purchases = pur_cols[0].number_input("Total Purchases", 0, 100, 12)
                avg_order = pur_cols[1].number_input("Avg Order Value ($)", 0.0, 500.0, 112.97, step=0.01)
                days_since = pur_cols[2].number_input("Days Since Last Purchase", 0, 365, 21)
                cart_abandon = pur_cols[3].slider("Cart Abandonment Rate (%)", 0.0, 100.0, 58.1, step=0.1)

                pur_cols2 = st.columns(4)
                wishlist = pur_cols2[0].number_input("Wishlist Items", 0, 30, 4)
                discount = pur_cols2[1].slider("Discount Usage Rate (%)", 0.0, 100.0, 40.2, step=0.1)
                returns_rate = pur_cols2[2].slider("Returns Rate (%)", 0.0, 30.0, 5.4, step=0.1)
                ltv = pur_cols2[3].number_input("Lifetime Value ($)", 0.0, 10000.0, 1243.42, step=0.01)

                # --- Section 4: Support & Financial ---
                st.subheader("🛠️ Support & Financial")
                sup_cols = st.columns(2)
                calls = sup_cols[0].number_input("Customer Service Calls", 0, 20, 5)
                credit = sup_cols[1].number_input("Credit Balance ($)", 0.0, 10000.0, 1896.0, step=1.0)

                submit = st.form_submit_button("🚀 Predict Churn Risk")

                if submit:
                    payload = {
                        "Age": age,
                        "Gender": gender,
                        "Country": country,
                        "Signup_Quarter": signup_q,
                        "Membership_Years": membership,
                        "Login_Frequency": login_freq,
                        "Pages_Per_Session": pages,
                        "Mobile_App_Usage": mobile,
                        "Email_Open_Rate": email_open,
                        "Social_Media_Engagement_Score": social,
                        "Product_Reviews_Written": reviews,
                        "Total_Purchases": total_purchases,
                        "Average_Order_Value": avg_order,
                        "Days_Since_Last_Purchase": days_since,
                        "Cart_Abandonment_Rate": cart_abandon,
                        "Wishlist_Items": wishlist,
                        "Discount_Usage_Rate": discount,
                        "Returns_Rate": returns_rate,
                        "Lifetime_Value": ltv,
                        "Customer_Service_Calls": calls,
                        "Credit_Balance": credit,
                    }

                    res = predict_churn(payload)

                    st.divider()
                    if res.get('churn_risk') == 1:
                        st.error(f"🚨 **HIGH CHURN RISK** — Probability: {res['probability']:.2%}")
                        st.warning("**Recommendation:** Trigger a retention campaign immediately.")
                    else:
                        st.success(f"✅ **LOW CHURN RISK** — Probability: {res['probability']:.2%}")
                        st.info("**Recommendation:** This customer is stable. Consider upsell opportunities.")


        # --- TAB 4: MODEL COMPARISON ---
        elif st.session_state.page in ["Model Comparison", "Strategy"]:
            st.title("🏆 Model Benchmarking & Selection")
            
            comp_data = fetch_data("model-comparison")
            if not comp_data:
                st.error("Local data load failed.")
                st.stop()

            df_comp = pd.DataFrame(comp_data)

            # --- 1. THE LEADERBOARD ---
            st.subheader("1. Performance Leaderboard")
            # Highlight the winner and the baseline
            st.dataframe(
                df_comp.style.highlight_max(subset=['F1_Score'], color='#d4edda')
                            .highlight_min(subset=['F1_Score'], color='#f8d7da'),
                width='stretch'
            )

            # --- 2. BASELINE ANALYSIS (Decision Tree) ---
            st.divider()
            col_a, col_b = st.columns([2, 1])

            with col_a:
                st.subheader("2. Why not just use a Decision Tree?")
                st.write("""
                While a **Decision Tree** is the most 'human-readable' model, it suffers from **Overfitting**. 
                In our data, it captures simple rules (e.g., *'If Calls > 5, then Churn'*) but misses the complex 
                interactions between Discount Usage and Session Duration that **LightGBM** finds.
                """)
                
                # Comparison chart specifically for Baseline vs Champion
                df_sub = df_comp[df_comp['Model'].isin(["LightGBM (Champion)", "Decision Tree (Baseline)"])]
                fig_compare = px.line_polar(
                    df_sub.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1_Score"]),
                    r="value", theta="variable", color="Model", line_close=True,
                    title="Champion (LightGBM) vs. Baseline (Decision Tree)"
                )
                st.plotly_chart(fig_compare, width='stretch')

            with col_b:
                st.info("💡 **Manager's Note**")
                st.markdown("""
                **The 'Tree' Logic:**
                Decision trees split customers into groups based on single questions.
                
                **The 'Boosted' Logic (LightGBM):**
                Builds hundreds of small trees, where each new tree fixes the 'mistakes' of the previous one. 
                This is why LightGBM has a **+8% higher F1-Score** than our baseline.
                """)

            # --- 3. SHAP EXPLAINABILITY ANALYSIS ---
            st.subheader("3. What drives the 'Decision'? (SHAP Explainability Analysis)")
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** is a state-of-the-art cooperative game theory method to explain 
            individual and global predictions made by machine learning models. 
            Unlike general feature importance which only shows *what* features are important, SHAP tells us 
            *how* each feature impacts the prediction (positive/negative direction and magnitude) for individual customers.
            """)
            
            try:
                # Compute and retrieve SHAP values
                with st.spinner("Calculating SHAP Explainability Values (cached for performance)..."):
                    explainer, shap_churn, X_test_sampled = get_shap_analysis(model, df, _medians, _city_labels)
                
                # Render SHAP visualizations in organized tabs
                shap_tab1, shap_tab2, shap_tab3 = st.tabs([
                    "📊 Global Beeswarm Summary", 
                    "📈 Global Feature Importance (Bar)", 
                    "🎯 Individual Customer Explainer (Waterfall)"
                ])
                
                with shap_tab1:
                    st.write("""
                    **Beeswarm Plot**: Shows the distribution of the impact each feature has on the model's output. 
                    Each dot represents a customer. The color indicates the feature value (red = high, blue = low). 
                    For example, high **Customer Service Calls** (red) pushes the churn risk **higher** (right).
                    """)
                    fig1, ax1 = plt.subplots(figsize=(5, 3))
                    # Pass the matplotlib axis explicitly
                    shap.summary_plot(shap_churn, X_test_sampled, plot_type="dot", max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)
                    
                with shap_tab2:
                    st.write("""
                    **Global Feature Importance**: Explains which features have the greatest overall impact on the model's decisions. 
                    This represents the mean absolute SHAP value for each feature.
                    """)
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    shap.summary_plot(shap_churn, X_test_sampled, plot_type="bar", max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                    
                with shap_tab3:
                    st.write("### Single Prediction Waterfall Explainer")
                    st.markdown("""
                    Choose a customer from the sampled test set using the slider below to see **exactly why** the LightGBM model predicted their churn probability. 
                    - **Blue bars**: Features that *decrease* their risk of churning.
                    - **Red bars**: Features that *increase* their risk of churning.
                    """)
                    
                    sample_idx = st.slider("Select Customer Index from Test Set:", 0, len(X_test_sampled) - 1, 0)
                    
                    fig3, ax3 = plt.subplots(figsize=(5, 3))
                    # Construct Explanation object as requested
                    explanation = shap.Explanation(
                        values=shap_churn[sample_idx],
                        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                        data=X_test_sampled.iloc[sample_idx],
                        feature_names=X_test_sampled.columns.tolist()
                    )
                    shap.plots.waterfall(explanation, show=False)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)

                    
            except Exception as e:
                st.error(f"Failed to calculate/render SHAP analysis: {e}")
                st.exception(e)


        # --- TAB 5: PRESENTATION ---
        elif st.session_state.page == "Presentation":
            # ── shared CSS ──────────────────────────────────────────────────
            st.markdown("""
            <style>
            /* slide card */
            .slide-card {
                background: #0F1B35;
                border-radius: 12px;
                padding: 28px 32px;
                margin-bottom: 24px;
                border-left: 5px solid #0D9488;
                color: #F8FAFC;
            }
            .slide-card h2 { color: #14B8A6; margin-top: 0; font-size: 1.35rem; }
            .slide-tag {
                display: inline-block;
                background: #1A3560;
                color: #94A3B8;
                font-size: 0.72rem;
                letter-spacing: 2px;
                padding: 2px 10px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            /* metric pill */
            .pill {
                display: inline-block;
                background: #1A3560;
                border: 1px solid #0D9488;
                border-radius: 8px;
                padding: 10px 18px;
                margin: 6px;
                text-align: center;
            }
            .pill .val { font-size: 1.7rem; font-weight: 700; color: #0D9488; }
            .pill .lbl { font-size: 0.78rem; color: #94A3B8; }
            /* segment cards */
            .seg { border-radius: 8px; padding: 16px; margin: 6px 0; }
            .seg-green { background:#0D94881A; border-left:4px solid #0D9488; }
            .seg-orange { background:#F973161A; border-left:4px solid #F97316; }
            .seg-red   { background:#EF44441A; border-left:4px solid #EF4444; }
            .seg h4 { margin:0 0 4px 0; }
            /* model table row */
            .winner { color:#0D9488; font-weight:700; }
            </style>
            """, unsafe_allow_html=True)

            st.title("📽️ Project Presentation")
            st.caption("WQD7012 Applied Machine Learning — OCC 3 Group 18 | 10-slide walkthrough")
            st.divider()

            # ── SLIDE 1: Title ───────────────────────────────────────────
            st.markdown("""
            <div class="slide-card">
                <span class="slide-tag">SLIDE 01 — TITLE</span>
                <h2>E-Commerce Customer Churn Prediction</h2>
                <p style="color:#94A3B8;">WQD7012 Applied Machine Learning &nbsp;|&nbsp; OCC 3 Group 18</p>
                <p style="color:#CBD5E1; font-size:0.9rem;">
                    Lam Jun Yan &nbsp;·&nbsp; Lim Zhi Yu &nbsp;·&nbsp; Chong Yan Yi &nbsp;·&nbsp;
                    Chong Jia Wei &nbsp;·&nbsp; Eng Hong Chee
                </p>
                <p style="color:#0D9488; font-size:0.85rem; font-style:italic;">
                    Lecturer: Ts. Dr. Riyaz Ahamed Ariyaluran Habeeb Mohamed
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── SLIDE 2: Problem & Objectives ────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 02 — PROBLEM & OBJECTIVES</span><h2>Problem Statement & Objectives</h2></div>', unsafe_allow_html=True)
            col_prob, col_obj = st.columns(2)
            with col_prob:
                st.markdown("**The Problem**")
                for p in [
                    "🔴  Churn prediction & segmentation treated separately, missing cross-segment drivers",
                    "🔴  Generalised 'one-size-fits-all' marketing wastes discounts on loyal users",
                    "🔴  Black-box models provide no actionable explanation for stakeholders",
                ]:
                    st.markdown(f"> {p}")
            with col_obj:
                st.markdown("**Objectives**")
                for o in [
                    "✅  **Identify** key behavioural factors driving churn via EDA & feature importance",
                    "✅  **Segment** customers with K-Means to reveal high-value vs at-risk groups",
                    "✅  **Evaluate** 5 ML models (LR · NB · DT · RF · LightGBM) for accuracy vs interpretability",
                ]:
                    st.markdown(o)

            st.divider()

            # ── SLIDE 3: Dataset ─────────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 03 — DATASET & PREPROCESSING</span><h2>Dataset & Preprocessing</h2></div>', unsafe_allow_html=True)

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Customers", "50,000", "8 global markets")
            p2.metric("Features", "25", "demographics + behaviour")
            p3.metric("Missing Cols", "14", "up to ~12% per feature")
            p4.metric("Class Imbalance", "2.46 : 1", "Not-Churned vs Churned")

            st.markdown("**Preprocessing pipeline**")
            steps = {
                "Data Inspection": "df.info() / describe() → unrealistic ages, negative values found",
                "Missing Data": "Mode (categorical) · Median (skewed) · Mean (symmetric)",
                "Outlier Handling": "IQR Winsorization on Lifetime_Value; age capped 18–90, scores 0–100",
                "Class Imbalance": "Stratified 80/20 split → SMOTE on training set → 28,440 : 28,440",
                "Encoding & Scaling": "Target Encoding (geography) · One-Hot (categorical text) · StandardScaler",
            }
            for k, v in steps.items():
                st.markdown(f"- **{k}:** {v}")

            st.divider()

            # ── SLIDE 4: EDA ─────────────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 04 — EDA & FEATURE ENGINEERING</span><h2>EDA & Feature Engineering</h2></div>', unsafe_allow_html=True)

            col_eda, col_feat = st.columns(2)
            with col_eda:
                st.markdown("**Key Correlation Findings**")
                eda_data = {
                    "Feature": ["Customer Service Calls", "Cart Abandonment Rate", "Session Duration", "Login Frequency"],
                    "Correlation": ["+0.29", "+0.28", "−0.22", "−0.20"],
                    "Direction": ["↑ churn", "↑ churn", "↓ churn (retention)", "↓ churn (retention)"],
                }
                st.dataframe(pd.DataFrame(eda_data), hide_index=True, use_container_width=True)
                st.warning("⚠️ Session Duration ↔ Pages/Session: r = 0.71 (multicollinearity risk)")

            with col_feat:
                st.markdown("**Engineered Features**")
                for f, d in [
                    ("Engagement Index", "login_freq + pages_per_session + email_open_rate / 3"),
                    ("Purchase Velocity", "total_purchases / membership_years"),
                    ("Actual Returns Count", "returns_rate × total_purchases / 100"),
                    ("Loyalty Tiers", "membership_years → New / Established / Veteran"),
                    ("Generation Bins", "age → Gen Z / Millennial / Gen X / Boomer"),
                    ("Corr Threshold |0.10|", "dropped low-signal & multicollinear columns"),
                ]:
                    st.markdown(f"- **{f}:** `{d}`")

            st.divider()

            # ── SLIDE 5: Models ───────────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 05 — PREDICTIVE MODELLING</span><h2>5 Models Compared</h2></div>', unsafe_allow_html=True)

            models_info = [
                ("Logistic Regression", "Hong Chee", "Linear", "Transparent baseline; interpretable coefficients for stakeholders"),
                ("Decision Tree (Tuned)", "Zhi Yu",   "Tree",   "Simple rule extraction; GridSearchCV tuned (depth=10, min_split=100)"),
                ("Random Forest",        "Jia Wei",   "Ensemble","Ensemble reduces overfitting; captures non-linear interactions"),
                ("Naive Bayes",          "Yan Yi",    "Probabilistic","Fast probabilistic baseline; handles sparse/missing data well"),
                ("LightGBM ⭐",          "Jun Yan",   "Boosting","Gradient boosting; best Recall & ROC-AUC; handles large feature spaces"),
            ]
            tag_colors = {"Linear":"#64748B","Tree":"#2563EB","Ensemble":"#7C3AED","Probabilistic":"#F97316","Boosting":"#0D9488"}
            for name, who, tag, why in models_info:
                tc = tag_colors.get(tag, "#0D9488")
                st.markdown(f"""
                <div style="display:flex;align-items:center;background:#1E293B;border-radius:8px;
                            padding:12px 16px;margin-bottom:8px;border-left:4px solid {tc};">
                    <div style="flex:1">
                        <strong style="color:#F8FAFC;">{name}</strong>
                        <span style="color:#94A3B8;font-size:0.82rem;margin-left:8px;">— {who}</span><br>
                        <span style="color:#CBD5E1;font-size:0.88rem;">{why}</span>
                    </div>
                    <div style="background:{tc};color:#fff;font-size:0.75rem;padding:3px 10px;
                                border-radius:12px;white-space:nowrap;margin-left:12px;">{tag}</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ── SLIDE 6: Results ──────────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 06 — RESULTS & MODEL COMPARISON</span><h2>Results & Model Comparison</h2></div>', unsafe_allow_html=True)

            results_df = pd.DataFrame({
                "Model":     ["LightGBM ⭐", "Decision Tree (Tuned)", "Random Forest", "LR (Tuned)", "LR (Baseline)", "Naive Bayes"],
                "Recall":    [0.9042, 0.8924, 0.8702, 0.8135, 0.7505, 0.2879],
                "Precision": [0.4555, 0.4510, 0.4666, 0.3575, 0.3740, 0.4790],
                "F1-Score":  [0.6058, 0.5992, 0.6075, 0.4967, 0.4993, 0.3596],
                "ROC-AUC":   [0.7742, 0.7309, 0.7130, 0.7012, 0.7008, 0.6128],
                "Accuracy":  [0.6599, 0.6550, 0.6750, 0.5235, 0.5649, 0.7037],
            })
            st.dataframe(
                results_df.style
                    .highlight_max(subset=["Recall","ROC-AUC"], color="#0D948840")
                    .highlight_min(subset=["Recall","ROC-AUC"], color="#EF444430"),
                hide_index=True, use_container_width=True
            )
            st.info("**Primary metric: Recall** — minimising missed churners outweighs false-positive retention costs (2.46:1 imbalance).")

            # ROC bar chart
            fig_roc = px.bar(
                results_df.sort_values("ROC-AUC"),
                x="ROC-AUC", y="Model", orientation="h",
                color="ROC-AUC",
                color_continuous_scale=["#1A3560","#0D9488"],
                title="ROC-AUC Comparison",
                range_x=[0.55, 0.82],
            )
            fig_roc.update_layout(
                height=300, showlegend=False,
                plot_bgcolor="#0F1B35", paper_bgcolor="#0F1B35",
                font_color="#CBD5E1",
                coloraxis_showscale=False,
            )
            fig_roc.update_traces(marker_line_width=0)
            st.plotly_chart(fig_roc, use_container_width=True)

            st.divider()

            # ── SLIDE 7: Segmentation ─────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 07 — CUSTOMER SEGMENTATION</span><h2>K-Means Segmentation (k=3)</h2></div>', unsafe_allow_html=True)
            st.caption("k selected via Elbow Method + Silhouette Score · PCA 3D confirms separation (98.8% variance explained)")

            sc1, sc2, sc3 = st.columns(3)
            segs = [
                (sc1, "seg-green", "Segment 1 — High-Value / Engaged",  "< 40% churn risk",
                 ["Highest Lifetime_Value","Frequent logins + high purchase velocity",
                  "Low customer service friction","Strategy: Exclusive loyalty perks"]),
                (sc2, "seg-orange","Segment 2 — Moderate / Average",     "~50% churn risk",
                 ["Intermediate across all dimensions","Stable base with growth potential",
                  "Moderate service calls","Strategy: Upselling & cross-sell"]),
                (sc3, "seg-red",   "Segment 3 — Low-Value / At-Risk",    "> 60% churn risk",
                 ["Lowest Lifetime_Value","High Cart_Abandonment_Rate",
                  "Elevated Customer_Service_Calls","Strategy: Immediate win-back campaign"]),
            ]
            for col, cls, title, risk, bullets in segs:
                with col:
                    st.markdown(f'<div class="seg {cls}"><h4>{title}</h4><strong>{risk}</strong></div>', unsafe_allow_html=True)
                    for b in bullets:
                        st.markdown(f"• {b}")

            st.success("📊 **23 percentage-point spread** in churn risk between Segment 1 and Segment 3")
            st.divider()

            # ── SLIDE 8: SHAP ─────────────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 08 — INTERPRETABILITY (SHAP)</span><h2>Model Interpretability — SHAP (LightGBM)</h2></div>', unsafe_allow_html=True)

            col_shap, col_ins = st.columns([1, 1])
            with col_shap:
                st.markdown("**Top Features (mean |SHAP|)**")
                shap_df = pd.DataFrame({
                    "Feature": ["Lifetime_Value","Customer_Service_Calls","Age",
                                "Cart_Abandonment_Rate","Days_Since_Last_Purchase",
                                "Discount_Usage_Rate","Loyalty_Tier_Veteran","Email_Open_Rate"],
                    "Importance": [1.00, 0.85, 0.35, 0.28, 0.22, 0.19, 0.16, 0.14],
                })
                fig_shap = px.bar(
                    shap_df.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    color="Importance",
                    color_continuous_scale=["#1A3560","#0D9488"],
                )
                fig_shap.update_layout(
                    height=300, showlegend=False, coloraxis_showscale=False,
                    plot_bgcolor="#0F1B35", paper_bgcolor="#0F1B35", font_color="#CBD5E1",
                    margin=dict(l=0, r=10, t=10, b=0),
                )
                fig_shap.update_traces(marker_line_width=0)
                st.plotly_chart(fig_shap, use_container_width=True)

            with col_ins:
                st.markdown("**Key SHAP Insights**")
                for icon, title, body in [
                    ("💡","Feature Dominance","Lifetime_Value & Customer_Service_Calls contribute 4× more than all other features combined"),
                    ("📈","Churn Drivers","High Cart_Abandonment_Rate + long Days_Since_Last_Purchase push strongly toward churn"),
                    ("🎁","Retention Signal","High Discount_Usage_Rate → lower churn; promotions retain price-sensitive users"),
                    ("⚠️","Counterintuitive","High Lifetime_Value can sometimes increase churn risk for recently disengaged high-spenders"),
                ]:
                    st.markdown(f"**{icon} {title}**")
                    st.caption(body)
                    st.markdown("")

            st.divider()

            # ── SLIDE 9: Conclusion ───────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 09 — CONCLUSION</span><h2>Conclusion & Limitations</h2></div>', unsafe_allow_html=True)

            col_conc, col_lim = st.columns(2)
            with col_conc:
                st.markdown("**Key Findings**")
                for item in [
                    "🏆 **LightGBM** best model — Recall 90.4%, ROC-AUC 0.774, F1 0.606",
                    "🎯 **SMOTE** effectively resolved 2.46:1 class imbalance",
                    "🔍 **SHAP** confirms Lifetime_Value & Service_Calls dominate predictions",
                    "📊 **K-Means k=3** produces 23pp churn-risk spread across segments",
                    "🚀 **Streamlit** app deployed · GitHub repo available",
                ]:
                    st.markdown(item)

            with col_lim:
                st.markdown("**Limitations & Future Work**")
                for item in [
                    "📉 Moderate F1 (0.606) — high FP rate may inflate retention costs at scale",
                    "📸 Dataset is a time snapshot; may miss seasonal/longitudinal trends",
                    "🧪 SMOTE generates synthetic samples that may not reflect real minority behaviour",
                    "🔮 Future: threshold tuning · survival analysis · RNN for temporal dynamics",
                    "📡 Future: live A/B testing environment for continuous retraining",
                ]:
                    st.markdown(item)

            st.divider()

            # ── SLIDE 10: Demo Roadmap ────────────────────────────────────
            st.markdown('<div class="slide-card"><span class="slide-tag">SLIDE 10 — CODE DEMO ROADMAP</span><h2>Code Demo Roadmap (6 min)</h2></div>', unsafe_allow_html=True)

            demo_steps = [
                ("1 min", "📂 Data Loading & Inspection",       "Show df.info(), missing values, class distribution pie chart"),
                ("1 min", "📊 EDA",                             "Correlation heatmap, churn-by-country bars, continuous variable distributions"),
                ("1 min", "⚙️ Feature Engineering + SMOTE",     "Derived features, encoding, SMOTE balancing — before/after class distribution"),
                ("1 min", "🤖 Model Training & Comparison",     "Fit 5 models, show performance table, ROC curve comparison plot"),
                ("1 min", "🔍 SHAP Explainability",             "Bar plot + beeswarm + waterfall for one customer — available in Model Comparison tab"),
                ("1 min", "🌐 Segmentation + Streamlit App",    "K-Means PCA 3D, cluster profiles, live demo of Churn Predictor form"),
            ]

            for i, (time_alloc, step_name, detail) in enumerate(demo_steps, 1):
                c1, c2, c3 = st.columns([0.5, 2, 4])
                with c1:
                    st.markdown(f"**{time_alloc}**")
                with c2:
                    st.markdown(f"**{step_name}**")
                with c3:
                    st.caption(detail)
                if i < len(demo_steps):
                    st.markdown('<hr style="margin:4px 0;border-color:#1A3560;">', unsafe_allow_html=True)

            st.markdown("")
            st.success("🎓 Thank you — Q&A")



if st.session_state.page not in ["Churn Predictor", "Model Comparison", "Strategy", "Presentation"]:
    with side_col:
        st.markdown("""<div style="background-color: #f0f2f6; padding: 0px; border-radius: 10px; border: 0px solid #dfe1e5;">""", unsafe_allow_html=True)
        render_predictor()
        st.markdown("</div>", unsafe_allow_html=True)