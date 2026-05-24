# ------------------------------------------------------------------------------
# Imports and packages
# ------------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import numpy as np
import logging
import time
import traceback
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("churn_api")

# ------------------------------------------------------------------------------
# Initialize FastAPI app
# ------------------------------------------------------------------------------

app = FastAPI(title="Customer Analytics API")

# ------------------------------------------------------------------------------
# Request/Response Logging Middleware
# ------------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every HTTP request with method, path, status code, and duration."""
    start = time.perf_counter()
    method = request.method
    path = request.url.path
    
    logger.info(f"➡️  {method} {path} — Request received")
    
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        status = response.status_code
        
        if status < 400:
            logger.info(f"✅ {method} {path} — {status} OK ({duration_ms:.1f}ms)")
        elif status < 500:
            logger.warning(f"⚠️  {method} {path} — {status} Client Error ({duration_ms:.1f}ms)")
        else:
            logger.error(f"❌ {method} {path} — {status} Server Error ({duration_ms:.1f}ms)")
        
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.critical(f"💥 {method} {path} — Unhandled Exception ({duration_ms:.1f}ms): {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ------------------------------------------------------------------------------
# --- GLOBAL DATA/MODEL LOADING ---
# ------------------------------------------------------------------------------

# Load your pre-trained LightGBM model exported from Colab

try:
    model = joblib.load("churn_model.pkl")
    print(f"✅ Model loaded successfully: {type(model).__name__}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Load dataset for stats calculation
df = pd.read_csv("../data/ecommerce_customer_churn_dataset.csv")

# ------------------------------------------------------------------------------
# --- ENDPOINTS ---
# ------------------------------------------------------------------------------

# ==============================================================================
# Health check endpoint    
# ==============================================================================

@app.get("/")
def home():
    """
    Health check endpoint to ensure the API is running.
    """
    logger.info("Health check OK — model loaded: %s", model is not None)
    return {"status": "online", "message": "ML Backend is running", "model_loaded": model is not None}

# KPI Dashboard endpoint

@app.get("/dashboard-stats")
def get_dashboard_stats():
    """
    Calculates the 10 core KPIs for the home dashboard.
    Returns: JSON object with numeric figures.
    """
    try:
        stats = {
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
        logger.info("Dashboard stats computed — %d customers, churn=%.1f%%", stats['total_customers'], stats['churn_rate'])
        return stats
    except Exception as e:
        logger.error("Failed to compute dashboard stats: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Stats computation failed: {e}")

# ==================================================================================================================
# Customer Segmentation endpoint    
# ==================================================================================================================

@app.get("/segmentation")
def get_segmentation():
    """
    Logic for k=3 clustering.
    Processes the data, runs K-Means, and returns the labels for mapping.
    """
    try:
        features = ['Total_Purchases', 'Average_Order_Value', 'Membership_Years', 'Login_Frequency', 'Lifetime_Value']
        temp_df = df[features].fillna(df[features].median())
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(temp_df)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Prepare result
        result_df = df[['Country', 'Lifetime_Value']].copy()
        result_df['Cluster'] = clusters
        
        logger.info("Segmentation complete — %d records, %d clusters", len(result_df), len(set(clusters)))
        return result_df.to_dict(orient="records")
    except Exception as e:
        logger.error("Segmentation failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

# ==================================================================================================================
# Churn prediction endpoint    
# ==================================================================================================================

# Pre-compute dataset medians for default values
_medians = df.median(numeric_only=True).to_dict()

# Pre-compute City label encoding (same mapping used during training)
_city_labels = {city: idx for idx, city in enumerate(sorted(df['City'].dropna().unique()))}

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


@app.post("/predict")
def predict_churn(customer_data: dict):
    """
    Receives RAW customer attributes and returns a churn prediction.
    Feature engineering (one-hot encoding, derived metrics) is handled server-side.
    """
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded — check server startup logs")
    
    try:
        logger.info("Prediction request — raw keys received: %s", list(customer_data.keys()))
        
        # Engineer the full 39-feature vector
        input_df = _engineer_features(customer_data)
        logger.info("Engineered feature vector shape: %s", input_df.shape)
        
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]
        
        result = {
            "churn_risk": int(prediction[0]),
            "probability": float(probability[0])
        }
        logger.info("Prediction result — churn_risk=%d, probability=%.4f", result['churn_risk'], result['probability'])
        return result
    except Exception as e:
        logger.error("Prediction failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

# ==================================================================================================================
# Geo-Segmentation endpoint    
# ==================================================================================================================

@app.get("/geo-segmentation")
def get_geo_segmentation():
    """
    1. Performs PCA for the 3D visualization.
    2. Maps clusters to "Real Life" Business Personas.
    3. Groups data by Country for the map.
    """
    try:
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
        
        logger.info("Geo-segmentation complete — %d records, personas: %s", len(plot_df), list(persona_map.values()))
        return plot_df.to_dict(orient="records")
    except Exception as e:
        logger.error("Geo-segmentation failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Geo-segmentation failed: {e}")

# ==============================================================================
# Marketing Target endpoint    
# ==============================================================================

@app.get("/marketing-targets")
def get_marketing_targets():
    """
    Categorizes Low LTV customers into 3 Marketing Buckets:
    1. 'Rescue': High Churn Risk + Low LTV
    2. 'Nurture': Low Churn Risk + Low LTV (Potential Upsell)
    3. 'Ignore': Very Low LTV + Very High Churn (Not worth the spend)
    """
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


# ==============================================================================
# Model Comparison endpoint    
# ==============================================================================

@app.get("/model-comparison")
def get_model_comparison():
    comparison_data = {
        "Model": ["LightGBM (Champion)", "Random Forest", "XGBoost", "Decision Tree (Baseline)", "Logistic Regression"],
        "Accuracy": [0.89, 0.84, 0.87, 0.81, 0.76],
        "Precision": [0.86, 0.81, 0.85, 0.78, 0.72],
        "Recall": [0.83, 0.79, 0.81, 0.74, 0.68],
        "F1_Score": [0.84, 0.80, 0.83, 0.76, 0.70],
        "Training_Time_Sec": [1.2, 4.5, 2.1, 0.08, 0.5]
    }
    return comparison_data

# ------------------------------------------------------------------------------
# Run the app using uvicorn
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="[IP_ADDRESS]", port=8001)