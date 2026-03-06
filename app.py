# app.py
import streamlit as st
import pandas as pd
from modules.ingestion import validate_file, load_data
from modules.preprocessing import safe_preprocess
from modules.analysis import (
    validate_data_for_analysis,
    run_complete_analysis,
    run_regression,
    run_clustering,
    run_anomaly_detection,
    run_forecasting
)
from logger import get_logger
import config

logger = get_logger("app")

st.set_page_config(page_title="Auto-Insight", layout="wide")

def main():
    st.title("🚀 Auto-Insight: The Automatic Data Analytics Engine")
    st.markdown("---")

    # Sidebar for file upload
    st.sidebar.header("📁 Data Ingestion")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset", 
        type=["csv", "xlsx", "json"]
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Validating and loading file..."):
                # 1. Validate file format and size
                is_valid, error_msg = validate_file(uploaded_file)
                
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return

                # 2. Load data
                df, status = load_data(uploaded_file)
                
                if df is None:
                    st.error(f"❌ {status}")
                    return
                
                st.success(f"✅ Successfully loaded '{uploaded_file.name}'")
            
            # ========== DATA PREVIEW SECTION ==========
            st.subheader("📊 Data Preview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric("Missing Data", f"{null_percentage:.1f}%")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            with st.expander("📋 Column Information"):
                st.write(df.dtypes)
            
            # ========== PREPROCESSING SECTION ==========
            st.subheader("🔧 Data Preprocessing")
            
            if st.button("Clean Data", key="preprocess_btn"):
                with st.spinner("Preprocessing data..."):
                    cleaned_df, audit = safe_preprocess(df)
                    
                    if len(cleaned_df) == 0:
                        st.error("❌ Preprocessing failed: Result is empty")
                        st.write("**Audit Report:**", audit)
                        return
                    
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.audit = audit
                    st.success("✅ Data preprocessing complete")
                    
                    # Show audit report
                    with st.expander("📝 Preprocessing Report"):
                        st.write(f"**Rows removed:** {audit.get('rows_removed_total', 0)}")
                        st.write(f"**Duplicates removed:** {audit.get('duplicate_rows_removed', 0)}")
                        st.write(f"**Missing values handled:** {len(audit.get('missing_values_handled', {}))}")
                        st.write(f"**Outliers detected:** {len(audit.get('outliers_detected', {}))}")
                        if audit.get('errors'):
                            st.warning("⚠️ Warnings: " + ", ".join(audit['errors']))
            
            # Check if cleaned data is available
            if 'cleaned_df' not in st.session_state:
                st.info("💡 Click 'Clean Data' to preprocess your data before running analysis")
                return
            
            cleaned_df = st.session_state.cleaned_df
            
            # Show cleaned data stats
            st.subheader("✨ Cleaned Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Clean Rows", len(cleaned_df))
            with col2:
                st.metric("Clean Columns", len(cleaned_df.columns))
            with col3:
                null_pct = (cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns)) * 100)
                st.metric("Remaining Missing", f"{null_pct:.1f}%")
            with col4:
                st.metric("Data Quality", f"{100 - null_pct:.1f}%")
            
            # ========== ANALYSIS ENGINE SECTION ==========
            st.subheader("🤖 Machine Learning Analysis Engine")
            
            # Select target column
            target_col = st.selectbox(
                "Select Target Column (for Regression)",
                [None] + list(cleaned_df.columns),
                format_func=lambda x: "None" if x is None else x
            )
            
            # RUN ENGINE BUTTON
            if st.button("🚀 Run Analysis Engine", key="run_engine"):
                with st.spinner("🔍 Running complete analysis (Regression, Clustering, Anomaly, Forecasting)..."):
                    
                    # ===== VALIDATION STEP =====
                    is_valid, validation_msg = validate_data_for_analysis(cleaned_df, target_col)
                    
                    if not is_valid:
                        st.error(f"❌ **Data Validation Failed**\n\n{validation_msg}")
                        st.info("💡 Please ensure your data meets the minimum requirements before running analysis.")
                        return
                    
                    st.success("✅ Data validation passed")
                    
                    # ===== RUN COMPLETE ANALYSIS =====
                    results = run_complete_analysis(cleaned_df, target_col)
                    
                    # Store results in session
                    st.session_state.analysis_results = results
                    
                    # ===== DISPLAY RESULTS =====
                    st.divider()
                    st.subheader("📈 Analysis Results")
                    
                    # Overall status
                    status_msg = results.get('summary', {}).get('status', 'Unknown status')
                    if results.get('has_errors') and results['summary'].get('tasks_completed', 0) > 0:
                        st.warning(f"⚠️ {status_msg}")
                    elif results.get('has_errors'):
                        st.error(f"❌ {status_msg}")
                    else:
                        st.success(f"✅ {status_msg}")
                    
                    # Validation errors
                    if 'validation_error' in results:
                        st.error(results['validation_error'])
                        return
                    
                    # Create tabs for each analysis
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📉 Regression",
                        "🎯 Clustering",
                        "🚨 Anomaly Detection",
                        "📊 Forecasting"
                    ])
                    
                    # REGRESSION TAB
                    with tab1:
                        reg_result = results.get('regression', {})
                        if reg_result.get('success'):
                            st.success("✅ Regression Analysis Completed")
                            
                            # Check for low R² score and show warning
                            r2_score = reg_result.get('r2_score', 0)
                            if r2_score < 0.3:
                                st.warning(
                                    "⚠️ Low model accuracy detected. Your data may contain quality issues. "
                                    "Please review missing values, outliers, or incorrect entries."
                                )
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("RMSE", f"{reg_result.get('metric_value', 0):.4f}")
                                st.metric("R² Score", f"{r2_score:.4f}")
                            with col2:
                                st.metric("MSE", f"{reg_result.get('mse', 0):.4f}")
                                st.metric("Test Samples", reg_result.get('samples_used', 0))
                            
                            if reg_result.get('feature_importance'):
                                st.subheader("Top 10 Feature Importance")
                                importance_data = list(reg_result['feature_importance'].items())[:10]
                                features = [f[0] for f in importance_data]
                                scores = [f[1] for f in importance_data]
                                st.bar_chart(dict(zip(features, scores)))
                        else:
                            st.warning(reg_result.get('error', '⚠️ Regression analysis skipped'))
                    
                    # CLUSTERING TAB
                    with tab2:
                        clust_result = results.get('clustering', {})
                        if clust_result.get('success'):
                            st.success("✅ Clustering Analysis Completed")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Silhouette Score", f"{clust_result.get('metric_value', 0):.4f}")
                                st.metric("Clusters Found", clust_result.get('n_clusters', 0))
                            with col2:
                                st.metric("Features Used", clust_result.get('features_used', 0))
                                st.metric("Samples Used", clust_result.get('samples_used', 0))
                            
                            if clust_result.get('cluster_distribution'):
                                st.subheader("Cluster Distribution")
                                st.bar_chart(clust_result['cluster_distribution'])
                        else:
                            st.warning(clust_result.get('error', '⚠️ Clustering analysis failed'))
                    
                    # ANOMALY DETECTION TAB
                    with tab3:
                        anom_result = results.get('anomaly_detection', {})
                        if anom_result.get('success'):
                            st.success("✅ Anomaly Detection Completed")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Anomalies Found", anom_result.get('anomalies_found', 0))
                                st.metric("Anomaly %", f"{anom_result.get('anomaly_percent', 0):.1f}%")
                            with col2:
                                st.metric("Normal Samples", anom_result.get('normal_samples', 0))
                                st.metric("Features Used", anom_result.get('features_used', 0))
                            
                            # Show as pie chart
                            anomaly_data = {
                                'Normal': anom_result.get('normal_samples', 0),
                                'Anomalous': anom_result.get('anomalies_found', 0)
                            }
                            st.subheader("Data Classification")
                            st.bar_chart(anomaly_data)
                        else:
                            st.warning(anom_result.get('error', '⚠️ Anomaly detection failed'))
                    
                    # FORECASTING TAB
                    with tab4:
                        fcst_result = results.get('forecasting', {})
                        if fcst_result.get('success'):
                            st.success("✅ Forecasting Analysis Completed")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Trend", f"{fcst_result.get('trend', 0):.4f}")
                                st.metric("Last Value", f"{fcst_result.get('last_value', 0):.2f}")
                            with col2:
                                st.metric("Volatility", f"{fcst_result.get('volatility', 0):.4f}")
                                st.metric("Next Period Forecast", f"{fcst_result.get('forecast_next_period', 0):.2f}")
                            
                            st.subheader("Statistics")
                            forecast_stats = {
                                "Last Value": fcst_result.get('last_value', 0),
                                "Recent Average": fcst_result.get('recent_average', 0),
                                "Forecast": fcst_result.get('forecast_next_period', 0)
                            }
                            st.metric("Recent Average", f"{fcst_result.get('recent_average', 0):.2f}")
                            st.info(f"Based on {fcst_result.get('data_points_used', 0)} historical data points")
                        else:
                            st.warning(fcst_result.get('error', '⚠️ Forecasting failed'))
                    
                    st.divider()
                    st.success("🎉 Analysis Complete! Review the results above.")
                                
        except Exception as e:
            logger.error(f"App error: {str(e)}")
            st.error(f"❌ An unexpected error occurred: {str(e)[:100]}")
            logger.error(f"Full error: {str(e)}")
    else:
        st.info("📤 Please upload a CSV, XLSX, or JSON file to begin analysis.")
        st.divider()
        st.markdown("""
        ### How to use Auto-Insight:
        1. **Upload** your dataset (CSV, XLSX, or JSON)
        2. **Clean** the data using the preprocessing button
        3. **Select** a target column (optional, for regression)
        4. **Run** the analysis engine to get insights from:
           - 📉 **Regression**: Predict numeric values
           - 🎯 **Clustering**: Group similar records
           - 🚨 **Anomaly Detection**: Find outliers
           - 📊 **Forecasting**: Predict trends
        """)

if __name__ == "__main__":
    main()
