import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st

# ==========================================
# LOGIC BREAKDOWN: DATA PARSING & SAFETY
# ==========================================
def parse_systolic_bp(bp_str):
    """Extracts systolic value from 'SBP/DBP' string format."""
    try:
        if isinstance(bp_str, str) and '/' in bp_str:
            return float(bp_str.split('/')[0])
        return float(bp_str) 
    except (ValueError, IndexError, TypeError):
        return np.nan

def train_mvp_model(df):
    """Preprocesses data and trains a robust Ridge Regression model."""
    df = df.copy()
    
    # 1. Pre-process Blood Pressure
    bp_col = 'Pre-Dialysis Blood Pressure'
    if bp_col in df.columns:
        df[bp_col] = df[bp_col].apply(parse_systolic_bp)

    # 2. Force Unify Sex Labels
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.strip().str.upper()
        df['Sex'] = df['Sex'].map({'M': 'Male', 'F': 'Female', 'MALE': 'Male', 'FEMALE': 'Female'}).fillna('Male')

    # 3. Clean Target Variable & Remove Extreme Outliers
    # Drops NaNs and filters out blatant clinical typos (e.g., a 50,000 IU accidental entry)
    df = df.dropna(subset=['Heparin'])
    df = df[(df['Heparin'] >= 500) & (df['Heparin'] <= 15000)]

    # 4. Target and Features
    y = df['Heparin']
    feature_cols = ['Age', 'Sex', 'Pre-Dialysis Weight', 'Target UF', 
                    'Treatment Duration', 'Vascular Access Type', 'Pre-Dialysis Blood Pressure']
    
    X = df[feature_cols].copy()
    
    # 5. Force ALL numeric features to be numeric
    numeric_features = ['Age', 'Pre-Dialysis Weight', 'Target UF', 'Treatment Duration', 'Pre-Dialysis Blood Pressure']
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    categorical_features = ['Sex', 'Vascular Access Type']

    # 6. Preprocessing pipelines (Added StandardScaler for Linear Model Stability)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 7. Full Model Pipeline (Swapped RF for Ridge)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(random_state=42))
    ])

    # 8. Set up Grid Search for Ridge Alpha penalty
    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
    }

    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 9. Feature Importance Mapping (Using Absolute Coefficients for Linear Model)
    ohe_features = model.named_steps['preprocessor'].transformers_[1][1]\
        .named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_features)
    
    # Extract coefficients and take absolute value for relative importance
    coefficients = model.named_steps['regressor'].coef_
    importances = np.abs(coefficients)
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

    return model, rmse, r2, importance_df, best_params

# ==========================================
# STREAMLIT UI COMPONENTS
# ==========================================
def run_app():
    st.set_page_config(page_title="Clinical Heparin Predictor", layout="centered")
    st.title("🩺 Heparin Bolus Dosage MVP")
    st.markdown("---")

    st.sidebar.header("Configuration")
    
    if "model" not in st.session_state:
        st.info("Pulling live data from Google Sheets and training the model...")
        url = "https://docs.google.com/spreadsheets/d/1_7IwgfcIZ_YOtpJiS-xMRdDmOPlnC3Nz1w3kogyPdCM/export?format=csv"
        
        try:
            df_real = pd.read_csv(url)
            
            # --- Robust Column Mapping ---
            mapping = {
                'Age/Sex': 'AgeSex',
                'Pre Dialysis Weight (kg)': 'Pre-Dialysis Weight',
                'Target UF (kg)': 'Target UF',
                'Total Duration of Treatment': 'DurationStr',
                'Access': 'Vascular Access Type',
                'BP (mmHg)': 'Pre-Dialysis Blood Pressure',
                'Heparin Bolus (IU)': 'Heparin'
            }
            df_real = df_real.rename(columns=mapping)

            # 1. Robust Age/Sex parsing
            def split_age_sex(val):
                if isinstance(val, str) and '/' in val:
                    parts = val.split('/')
                    return parts[0].strip(), parts[1].strip()
                return np.nan, np.nan
            
            df_real[['Age', 'Sex']] = df_real['AgeSex'].apply(lambda x: pd.Series(split_age_sex(x)))
            
            # 2. Robust Duration parsing
            def parse_duration(val):
                if isinstance(val, str):
                    nums = pd.Series(val).str.extract(r'(\d+)').iloc[0,0]
                    return float(nums) if pd.notnull(nums) else np.nan
                return val
            df_real['Treatment Duration'] = df_real['DurationStr'].apply(parse_duration)

            # 3. Clean Heparin
            df_real['Heparin'] = df_real['Heparin'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

            # Train
            model, rmse, r2, importance_df, best_params = train_mvp_model(df_real)
            
            st.session_state.model = model
            st.session_state.metrics = (rmse, r2)
            st.session_state.importance = importance_df
            st.session_state.best_params = best_params
            st.success("Model trained successfully on live data!")
            
        except Exception as e:
            st.error(f"Error loading or processing data: {e}")
            st.stop()

    # UI INPUTS
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (Years)", 18, 110, 60)
        sex = st.selectbox("Sex", ["Male", "Female"])
        pre_weight = st.slider("Pre-Dialysis Weight (kg)", 40.0, 150.0, 75.0)
        dur = st.slider("Treatment Duration (Hours)", 1.0, 6.0, 4.0)

    with col2:
        target_uf = st.slider("Target UF (Liters)", 0.0, 6.0, 2.5)
        v_access = st.selectbox("Vascular Access Type", ["AV Fistula", "Graft", "Catheter"])
        sbp = st.text_input("Pre-Dialysis BP (e.g., 140/90)", "130/85")

    # Prediction Logic
    input_df = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Pre-Dialysis Weight': [pre_weight],
        'Target UF': [target_uf],
        'Treatment Duration': [dur],
        'Vascular Access Type': [v_access],
        'Pre-Dialysis Blood Pressure': [sbp]
    })
    
    input_df['Pre-Dialysis Blood Pressure'] = input_df['Pre-Dialysis Blood Pressure'].apply(parse_systolic_bp)

    if st.button("Calculate Dosage"):
        prediction = st.session_state.model.predict(input_df)[0]
        
        # CLINICAL SAFETY GUARDRAIL
        final_dosage = min(max(prediction, 0.0), 5000.0) # Ensure no negative predictions
        
        st.subheader("Results")
        if prediction > 5000:
            st.warning("⚠️ Predicted dosage exceeded safety limit. Capped at 5,000 units.")
        elif prediction < 0:
            st.warning("⚠️ Model predicted negative dosage. Capped at 0 units. (Check data quality)")
        
        st.metric(label="Recommended Heparin Bolus", value=f"{int(final_dosage)} Units")
        
        st.markdown("---")
        st.markdown("### Model Diagnostics (Tuned)")
        rmse, r2 = st.session_state.metrics
        st.write(f"**RMSE:** {rmse:.2f} | **R² Score:** {r2:.2f}")
        
        st.markdown("#### Feature Importance (Absolute Coefficient Impact)")
        st.bar_chart(st.session_state.importance.set_index('feature'))

if __name__ == "__main__":
    run_app()