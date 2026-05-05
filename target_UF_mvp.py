import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st

# ==========================================
# LOGIC BREAKDOWN: DATA PARSING & SAFETY
# ==========================================
def parse_bp_systolic(bp_str):
    """Extracts systolic (top) value from 'SBP/DBP'."""
    try:
        if isinstance(bp_str, str) and '/' in bp_str:
            return float(bp_str.split('/')[0])
        return float(bp_str)
    except (ValueError, IndexError, TypeError):
        return np.nan

def parse_bp_diastolic(bp_str):
    """Extracts diastolic (bottom) value from 'SBP/DBP'."""
    try:
        if isinstance(bp_str, str) and '/' in bp_str:
            return float(bp_str.split('/')[1])
        return np.nan
    except (ValueError, IndexError, TypeError):
        return np.nan

def train_uf_model(df):
    """Preprocesses data and trains the RandomForest model for Target UF."""
    df = df.copy()
    
    # --- Robust Column Mapping to Match Google Sheet ---
    mapping = {
        'Age/Sex': 'AgeSex',
        'Pre Dialysis Weight (kg)': 'Pre-Dialysis Weight',
        'Dry Weight (kg)': 'Dry Weight',
        'BP (mmHg)': 'Pre-Dialysis Blood Pressure',
        'HR (min)': 'Pre-Dialysis Heart Rate',
        'Target UF (kg)': 'Target UF'
    }
    df = df.rename(columns=mapping)

    # 1. Parse Age and Sex
    def split_age_sex(val):
        if isinstance(val, str) and '/' in val:
            parts = val.split('/')
            return parts[0].strip(), parts[1].strip()
        return np.nan, np.nan
    
    if 'AgeSex' in df.columns:
        df[['Age', 'Sex']] = df['AgeSex'].apply(lambda x: pd.Series(split_age_sex(x)))
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # 2. Force Unify Sex Labels
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].astype(str).str.strip().str.upper()
        df['Sex'] = df['Sex'].map({'M': 'Male', 'F': 'Female', 'MALE': 'Male', 'FEMALE': 'Female'}).fillna('Male')

    # 3. Parse Blood Pressure into two separate features
    bp_col = 'Pre-Dialysis Blood Pressure'
    if bp_col in df.columns:
        df['Pre-Dialysis SBP'] = df[bp_col].apply(parse_bp_systolic)
        df['Pre-Dialysis DBP'] = df[bp_col].apply(parse_bp_diastolic)
    else:
        df['Pre-Dialysis SBP'] = np.nan
        df['Pre-Dialysis DBP'] = np.nan

    # 4. Clean the Target Variable (Target UF)
    # Ensure it's numeric and drop rows where the target is missing
    target_col = 'Target Ultrafiltration (UF)' if 'Target Ultrafiltration (UF)' in df.columns else 'Target UF'
    if target_col not in df.columns:
        # Fallback if the exact name varies
        potential_targets = [c for c in df.columns if 'target uf' in c.lower() or 'ultrafiltration' in c.lower()]
        if potential_targets:
            target_col = potential_targets[0]
        else:
            # Creation of dummy target to avoid crash if totally missing (unlikely)
            df['Target UF Num'] = np.nan
            target_col = 'Target UF Num'
    
    df['Target UF Num'] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=['Target UF Num'])

    y = df['Target UF Num']

    # 5. Define Features (Strictly Pre-Dialysis to prevent data leakage)
    # >>> NEW: Feature Engineering (The Math Fix) <<<
    df['Pre-Dialysis Weight'] = pd.to_numeric(df['Pre-Dialysis Weight'], errors='coerce')
    df['Dry Weight'] = pd.to_numeric(df['Dry Weight'], errors='coerce')
    df['Weight_Difference'] = df['Pre-Dialysis Weight'] - df['Dry Weight']

    feature_cols = ['Age', 'Sex', 'Pre-Dialysis Weight', 'Dry Weight', 'Weight_Difference',
                    'Pre-Dialysis SBP', 'Pre-Dialysis DBP', 'Pre-Dialysis Heart Rate']
    
    # Ensure all required columns exist in df
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    X = df[feature_cols].copy()
    
    # Force numeric conversion for safe processing
    numeric_features = ['Age', 'Pre-Dialysis Weight', 'Dry Weight', 'Weight_Difference', 'Pre-Dialysis SBP', 'Pre-Dialysis DBP', 'Pre-Dialysis Heart Rate']
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    categorical_features = ['Sex']

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
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

    # Full Model Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter Grid tuned for regression
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [3, 5, 7, 10],
        'regressor__min_samples_split': [2, 4, 8],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    # Grid Search with Cross-Validation
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
    
    ohe_features = model.named_steps['preprocessor'].transformers_[1][1]\
        .named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_features)
    importances = model.named_steps['regressor'].feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

    return model, rmse, r2, importance_df, best_params

# ==========================================
# STREAMLIT UI COMPONENTS
# ==========================================
def run_app():
    st.set_page_config(page_title="Target UF Predictor", layout="centered")
    st.title("Target UF Prediction MVP")
    st.markdown("---")

    st.sidebar.header("Configuration")
    
    if "model" not in st.session_state:
        # st.info("Pulling live data from Google Sheets and training the model...")
        url = "https://docs.google.com/spreadsheets/d/1_7IwgfcIZ_YOtpJiS-xMRdDmOPlnC3Nz1w3kogyPdCM/export?format=csv"
        
        try:
            df_real = pd.read_csv(url)
            model, rmse, r2, importance_df, best_params = train_uf_model(df_real)
            
            st.session_state.model = model
            st.session_state.metrics = (rmse, r2)
            st.session_state.importance = importance_df
            st.session_state.best_params = best_params
            # st.success("Model trained successfully on live data!")
            
        except Exception as e:
            st.error(f"Error loading or processing data: {e}")
            st.stop()

    # UI INPUTS
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (Years)", 18, 160 ,60)
        sex = st.selectbox("Sex", ["Male", "Female"])
        pre_weight = st.number_input("Pre-Dialysis Weight (kg)", 40.0, 150.0, 75.0, step=0.1)
        dry_weight = st.number_input("Target Dry Weight (kg)", 40.0, 150.0, 72.0, step=0.1)

    with col2:
        sbp = st.number_input("Pre-Dialysis Systolic BP", 70, 250, 130)
        dbp = st.number_input("Pre-Dialysis Diastolic BP", 40, 150, 80)
        hr = st.number_input("Pre-Dialysis Heart Rate", 40, 150, 75)

    # >>> NEW: Calculate the difference before sending to model <<<
    weight_diff = pre_weight - dry_weight

    # Inference Data Assembly
    input_df = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Pre-Dialysis Weight': [pre_weight],
        'Dry Weight': [dry_weight],
        'Weight_Difference': [weight_diff],  # Pass the engineered feature
        'Pre-Dialysis SBP': [sbp],
        'Pre-Dialysis DBP': [dbp],
        'Pre-Dialysis Heart Rate': [hr]
    })

    if st.button("Calculate Target UF"):
        
        # >>> NEW: CLINICAL HARD-STOP LOGIC <<<
        if weight_diff <= 0.1:
            final_uf = 0.0
            st.subheader("Results")
            st.success("🟢 Patient is at or below Target Dry Weight.")
            st.metric(label="Recommended Target UF", value=f"{final_uf:.2f} Liters")
            st.info("Clinical Override applied: Ultrafiltration is not required unless fluid administration is planned.")
            
        else:
            # Let the AI predict if there is actual weight gain
            prediction = st.session_state.model.predict(input_df)[0]
            
            # CLINICAL SAFETY GUARDRAIL: Cap UF at 5.0 Liters to prevent severe volume depletion
            final_uf = min(prediction, 5.0)
            final_uf = max(final_uf, 0.0) # Cannot be negative
            
            st.subheader("Results")
            if prediction > 5.0:
                st.warning("⚠️ Predicted Target UF exceeded safety limit. Capped at 5.0 Liters.")
            
            st.metric(label="Recommended Target UF", value=f"{final_uf:.2f} Liters")
        
        st.markdown("---")
        st.markdown("### Model Diagnostics (Tuned)")
        rmse, r2 = st.session_state.metrics
        st.write(f"**RMSE:** {rmse:.2f} | **R² Score:** {r2:.2f}")
        
        st.markdown("#### Feature Importance")
        st.bar_chart(st.session_state.importance.set_index('feature'))

if __name__ == "__main__":
    run_app()