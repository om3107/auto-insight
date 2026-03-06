# modules/analysis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score, r2_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Tuple, Optional
from logger import get_logger
from config import RANDOM_SEED

logger = get_logger("analysis")

# User-friendly error messages mapping
ERROR_MESSAGES = {
    "insufficient_rows": "Not enough clean data (minimum 20 rows required)",
    "insufficient_columns": "Not enough features (minimum 3 columns required)",
    "invalid_target": "Target column must contain numeric data",
    "no_numeric_features": "No numeric features available for analysis",
    "empty_dataframe": "Data is empty after preprocessing",
    "target_col_missing": "Target column not found in data",
    "constant_target": "Target column has no variation (all same value)",
    "all_nulls": "Column contains only null values",
    "insufficient_variance": "Data has insufficient variance for clustering",
    "singular_matrix": "Cannot compute due to singular data matrix",
}


def validate_data_for_analysis(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validates data before running ML analysis.
    
    Checks:
    - Minimum 20 rows
    - Target column is numeric (if provided)
    - At least 3 columns
    - DataFrame is not empty
    - Valid CSV data
    
    Args:
        df: Input DataFrame
        target_col: Optional target column name
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Check if dataframe is empty
        if df is None or len(df) == 0:
            return False, "❌ " + ERROR_MESSAGES["empty_dataframe"]
        
        # Check minimum rows
        if len(df) < 20:
            return False, f"❌ Not enough data: {len(df)} rows found, minimum 20 required"
        
        # Check minimum columns
        if len(df.columns) < 3:
            return False, f"❌ Not enough features: {len(df.columns)} columns found, minimum 3 required"
        
        # Check target column if provided
        if target_col:
            if target_col not in df.columns:
                return False, f"❌ Target column '{target_col}' not found in data"
            
            # Check if target is numeric or convertible
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                # Try to convert
                try:
                    pd.to_numeric(df[target_col], errors='coerce')
                    if df[target_col].isnull().all():
                        return False, f"❌ Target column '{target_col}' contains no numeric values"
                except:
                    return False, f"❌ Target column '{target_col}' must be numeric"
            
            # Check if target has variation
            if df[target_col].nunique() <= 1:
                return False, f"❌ Target column has no variation (all rows have same value)"
        
        # Check for at least some numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return False, "❌ No numeric columns available for analysis"
        
        logger.info(f"Data validation passed: {len(df)} rows, {len(df.columns)} columns")
        return True, ""
    
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, f"❌ Data validation failed: {str(e)}"


def fill_nulls_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silently fills all null values in a DataFrame.
    - Numeric columns: filled with median
    - Categorical columns: filled with mode (most frequent value)
    Never crashes, never throws errors.
    
    Args:
        df: Input DataFrame with possible nulls
        
    Returns:
        DataFrame with all nulls filled
    """
    try:
        filled_df = df.copy()
        
        # Get null counts before filling
        null_counts = filled_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()
        
        if len(cols_with_nulls) > 0:
            logger.info(f"Filling {len(cols_with_nulls)} columns with null values")
        
        for col in filled_df.columns:
            if filled_df[col].isnull().sum() == 0:
                continue  # Skip columns with no nulls
            
            try:
                # Check if numeric
                if pd.api.types.is_numeric_dtype(filled_df[col]):
                    # Fill numeric with median
                    median_val = filled_df[col].median()
                    if pd.notna(median_val):
                        filled_df[col] = filled_df[col].fillna(median_val)
                        logger.debug(f"Filled {col} with median: {median_val}")
                    else:
                        # If median is NaN, use 0
                        filled_df[col] = filled_df[col].fillna(0)
                        logger.debug(f"Filled {col} with 0 (median was NaN)")
                else:
                    # Fill categorical with mode
                    mode_val = filled_df[col].mode()
                    if len(mode_val) > 0:
                        fill_value = mode_val.iloc[0]
                    else:
                        # If no mode, use 'Unknown'
                        fill_value = 'Unknown'
                    
                    filled_df[col] = filled_df[col].fillna(fill_value)
                    logger.debug(f"Filled {col} with mode/frequent value: {fill_value}")
            
            except Exception as col_err:
                logger.warn(f"Could not fill column {col}: {str(col_err)}")
                # Try to fill with default
                if pd.api.types.is_numeric_dtype(filled_df[col]):
                    filled_df[col] = filled_df[col].fillna(0)
                else:
                    filled_df[col] = filled_df[col].fillna('Unknown')
        
        return filled_df
    
    except Exception as e:
        logger.warn(f"Null filling error: {str(e)}, returning original DataFrame")
        return df


def run_regression(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Safely runs regression analysis.
    Never crashes — returns error dict on failure.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Target column name
        
    Returns:
        Dict with results or error message
    """
    try:
        logger.info(f"Starting Regression on {target_col}")
        
        # Fill any remaining nulls (safety net)
        df = fill_nulls_safe(df)
        
        # Validate inputs
        if target_col not in df.columns:
            return {
                "success": False,
                "error": f"⚠️ Regression failed: {ERROR_MESSAGES['target_col_missing']}"
            }
        
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove rows with null target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(y) < 10:
            return {
                "success": False,
                "error": f"⚠️ Regression failed: Insufficient clean target values ({len(y)} found, need 10)"
            }
        
        # Encode categorical features
        try:
            X = pd.get_dummies(X, drop_first=True)
        except Exception as e:
            logger.warn(f"Could not encode features: {str(e)}")
            return {
                "success": False,
                "error": f"⚠️ Regression failed: Could not process categorical features"
            }
        
        if len(X.columns) == 0:
            return {
                "success": False,
                "error": f"⚠️ Regression failed: No features available after processing"
            }
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_SEED
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"⚠️ Regression failed: Could not split data - {str(e)}"
            }
        
        # Train model
        try:
            model = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=RANDOM_SEED)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)
        except Exception as e:
            logger.error(f"Regression training failed: {str(e)}")
            return {
                "success": False,
                "error": f"⚠️ Regression failed: Model training error - {str(e)[:50]}"
            }
        
        logger.info(f"Regression successful: RMSE = {rmse:.4f}, R² = {r2:.4f}")
        
        return {
            "success": True,
            "task": "Regression",
            "metric_name": "RMSE",
            "metric_value": float(rmse),
            "mse": float(mse),
            "r2_score": float(r2),
            "feature_importance": dict(sorted(
                zip(X.columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # Top 10 features
            "samples_used": len(X_test),
            "features_used": len(X.columns)
        }
    
    except Exception as e:
        logger.error(f"UNEXPECTED Regression error: {str(e)}")
        return {
            "success": False,
            "error": f"⚠️ Regression failed: Unexpected error - {str(e)[:50]}"
        }


def run_clustering(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely runs K-Means clustering analysis.
    Never crashes — returns error dict on failure.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Optional target column to exclude
        
    Returns:
        Dict with results or error message
    """
    try:
        logger.info("Starting Clustering")
        
        # Fill any remaining nulls (safety net)
        df = fill_nulls_safe(df)
        
        # Prepare features
        X = df.copy()
        if target_col and target_col in X.columns:
            X = X.drop(columns=[target_col])
        
        # Keep only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) == 0:
            return {
                "success": False,
                "error": "⚠️ Clustering failed: No numeric features available"
            }
        
        if len(X_numeric) < 10:
            return {
                "success": False,
                "error": f"⚠️ Clustering failed: Not enough samples ({len(X_numeric)} found, need 10)"
            }
        
        # Remove rows with all missing values
        X_numeric = X_numeric.dropna(how='all')
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        if len(X_numeric) < 10:
            return {
                "success": False,
                "error": "⚠️ Clustering failed: Not enough valid data points"
            }
        
        # Determine optimal number of clusters
        try:
            n_clusters = min(5, max(2, len(X_numeric) // 10))  # 2-5 clusters
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_SEED)
            clusters = kmeans.fit_predict(X_numeric)
            silhouette = silhouette_score(X_numeric, clusters)
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return {
                "success": False,
                "error": f"⚠️ Clustering failed: {str(e)[:50]}"
            }
        
        # Cluster distribution
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_distribution = {f"Cluster {int(k)}": int(v) for k, v in zip(unique, counts)}
        
        logger.info(f"Clustering successful: {n_clusters} clusters, Silhouette = {silhouette:.4f}")
        
        return {
            "success": True,
            "task": "Clustering",
            "metric_name": "Silhouette Score",
            "metric_value": float(silhouette),
            "n_clusters": n_clusters,
            "cluster_distribution": cluster_distribution,
            "features_used": len(X_numeric.columns),
            "samples_used": len(X_numeric)
        }
    
    except Exception as e:
        logger.error(f"UNEXPECTED Clustering error: {str(e)}")
        return {
            "success": False,
            "error": f"⚠️ Clustering failed: Unexpected error - {str(e)[:50]}"
        }


def run_anomaly_detection(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely runs Isolation Forest anomaly detection.
    Never crashes — returns error dict on failure.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Optional target column to exclude
        
    Returns:
        Dict with results or error message
    """
    try:
        logger.info("Starting Anomaly Detection")
        
        # Fill any remaining nulls (safety net)
        df = fill_nulls_safe(df)
        
        # Prepare features
        X = df.copy()
        if target_col and target_col in X.columns:
            X = X.drop(columns=[target_col])
        
        # Keep only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) == 0:
            return {
                "success": False,
                "error": "⚠️ Anomaly Detection failed: No numeric features"
            }
        
        if len(X_numeric) < 10:
            return {
                "success": False,
                "error": f"⚠️ Anomaly Detection failed: Not enough samples ({len(X_numeric)})"
            }
        
        # Remove rows with all missing
        X_numeric = X_numeric.dropna(how='all')
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        if len(X_numeric) < 10:
            return {
                "success": False,
                "error": "⚠️ Anomaly Detection failed: Not enough valid data"
            }
        
        # Run Isolation Forest
        try:
            iso_forest = IsolationForest(
                contamination=0.1,  # Assume 10% anomalies
                n_estimators=100,
                random_state=RANDOM_SEED
            )
            anomalies = iso_forest.fit_predict(X_numeric)
            anomaly_scores = iso_forest.score_samples(X_numeric)
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {
                "success": False,
                "error": f"⚠️ Anomaly Detection failed: {str(e)[:50]}"
            }
        
        # Calculate statistics
        n_anomalies = (anomalies == -1).sum()
        anomaly_percent = (n_anomalies / len(anomalies)) * 100
        
        logger.info(f"Anomaly Detection successful: {n_anomalies} anomalies ({anomaly_percent:.1f}%)")
        
        return {
            "success": True,
            "task": "Anomaly Detection",
            "metric_name": "Anomaly Percentage",
            "metric_value": float(anomaly_percent),
            "anomalies_found": int(n_anomalies),
            "normal_samples": int(len(anomalies) - n_anomalies),
            "anomaly_percent": float(anomaly_percent),
            "features_used": len(X_numeric.columns),
            "samples_used": len(X_numeric)
        }
    
    except Exception as e:
        logger.error(f"UNEXPECTED Anomaly Detection error: {str(e)}")
        return {
            "success": False,
            "error": f"⚠️ Anomaly Detection failed: Unexpected error - {str(e)[:50]}"
        }


def run_forecasting(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Safely runs time-series like forecasting analysis.
    Uses sequential pattern detection as a simple forecast.
    Never crashes — returns error dict on failure.
    
    Args:
        df: Preprocessed DataFrame (should have time order)
        target_col: Optional target column to forecast
        
    Returns:
        Dict with results or error message
    """
    try:
        logger.info("Starting Forecasting")
        
        # Fill any remaining nulls (safety net)
        df = fill_nulls_safe(df)
        
        if target_col is None:
            # Use first numeric column as target
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {
                    "success": False,
                    "error": "⚠️ Forecasting failed: No numeric columns to forecast"
                }
            target_col = numeric_cols[0]
        
        if target_col not in df.columns:
            return {
                "success": False,
                "error": f"⚠️ Forecasting failed: Column '{target_col}' not found"
            }
        
        y = df[target_col].dropna()
        
        if len(y) < 10:
            return {
                "success": False,
                "error": f"⚠️ Forecasting failed: Not enough data points ({len(y)}, need 10)"
            }
        
        try:
            # Simple trend analysis using differences
            values = np.asarray(y, dtype=float)
            
            # Calculate trend
            diff = np.diff(values)
            trend = np.mean(diff) if len(diff) > 0 else 0
            
            # Calculate volatility
            volatility = np.std(diff) if len(diff) > 0 else 0
            
            # Simple forecast: last value + trend
            last_value = float(values[-1])
            next_value = last_value + trend
            
            # Calculate recent average
            recent_avg = np.mean(values[-min(5, len(values)):])
            
        except Exception as e:
            logger.error(f"Forecast calculation failed: {str(e)}")
            return {
                "success": False,
                "error": f"⚠️ Forecasting failed: Calculation error"
            }
        
        logger.info(f"Forecasting successful: Trend = {trend:.4f}, Volatility = {volatility:.4f}")
        
        return {
            "success": True,
            "task": "Forecasting",
            "metric_name": "Trend",
            "metric_value": float(trend),
            "last_value": last_value,
            "forecast_next_period": next_value,
            "trend": float(trend),
            "volatility": float(volatility),
            "recent_average": float(recent_avg),
            "data_points_used": len(values)
        }
    
    except Exception as e:
        logger.error(f"UNEXPECTED Forecasting error: {str(e)}")
        return {
            "success": False,
            "error": f"⚠️ Forecasting failed: Unexpected error - {str(e)[:50]}"
        }


def run_complete_analysis(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    MAIN ENGINE: Runs ALL ML tasks independently.
    One failure does NOT stop others.
    Always returns results (success or error) for each task.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Optional target column for supervised tasks
        
    Returns:
        Dict with results from all tasks (successful or failed)
    """
    
    logger.info("=" * 60)
    logger.info("STARTING COMPLETE ML ANALYSIS ENGINE")
    logger.info("=" * 60)
    
    results = {
        "has_errors": False,
        "validation_passed": False,
        "regression": {"success": False},
        "clustering": {"success": False},
        "anomaly_detection": {"success": False},
        "forecasting": {"success": False},
        "summary": {}
    }
    
    try:
        # ========== STEP 1: VALIDATE DATA ==========
        logger.info("\n[1/5] VALIDATING DATA")
        is_valid, validation_msg = validate_data_for_analysis(df, target_col)
        
        if not is_valid:
            logger.error(f"Validation failed: {validation_msg}")
            results["has_errors"] = True
            results["validation_error"] = validation_msg
            results["summary"]["status"] = f"❌ {validation_msg}"
            return results
        
        results["validation_passed"] = True
        logger.info(f"✓ Data validation passed")
        
        # ========== STEP 1.5: FILL NULL VALUES SILENTLY ==========
        logger.info("\n[1.5/5] FILLING NULL VALUES")
        df = fill_nulls_safe(df)
        logger.info("✓ Null values handled")
        
        # ========== STEP 2: REGRESSION (Independent Task) ==========
        logger.info("\n[2/5] RUNNING REGRESSION")
        if target_col:
            results["regression"] = run_regression(df, target_col)
            if results["regression"]["success"]:
                logger.info("✓ Regression completed successfully")
            else:
                logger.warn(f"✗ Regression failed: {results['regression'].get('error', 'Unknown error')}")
                results["has_errors"] = True
        else:
            results["regression"] = {
                "success": False,
                "error": "⚠️ Regression skipped: No target column specified"
            }
        
        # ========== STEP 3: CLUSTERING (Independent Task) ==========
        logger.info("\n[3/5] RUNNING CLUSTERING")
        results["clustering"] = run_clustering(df, target_col)
        if results["clustering"]["success"]:
            logger.info("✓ Clustering completed successfully")
        else:
            logger.warn(f"✗ Clustering failed: {results['clustering'].get('error', 'Unknown error')}")
            results["has_errors"] = True
        
        # ========== STEP 4: ANOMALY DETECTION (Independent Task) ==========
        logger.info("\n[4/5] RUNNING ANOMALY DETECTION")
        results["anomaly_detection"] = run_anomaly_detection(df, target_col)
        if results["anomaly_detection"]["success"]:
            logger.info("✓ Anomaly Detection completed successfully")
        else:
            logger.warn(f"✗ Anomaly Detection failed: {results['anomaly_detection'].get('error', 'Unknown error')}")
            results["has_errors"] = True
        
        # ========== STEP 5: FORECASTING (Independent Task) ==========
        logger.info("\n[5/5] RUNNING FORECASTING")
        results["forecasting"] = run_forecasting(df, target_col)
        if results["forecasting"]["success"]:
            logger.info("✓ Forecasting completed successfully")
        else:
            logger.warn(f"✗ Forecasting failed: {results['forecasting'].get('error', 'Unknown error')}")
            results["has_errors"] = True
        
        # ========== FINAL SUMMARY ==========
        successful_tasks = sum(1 for task in [
            results["regression"], 
            results["clustering"],
            results["anomaly_detection"],
            results["forecasting"]
        ] if task.get("success", False))
        
        total_tasks = 4
        results["summary"]["tasks_completed"] = successful_tasks
        results["summary"]["tasks_total"] = total_tasks
        
        if results["has_errors"] and successful_tasks > 0:
            results["summary"]["status"] = f"⚠️ Partial results: {successful_tasks}/{total_tasks} analyses completed"
        elif successful_tasks == total_tasks:
            results["summary"]["status"] = "✅ All analyses completed successfully"
        else:
            results["summary"]["status"] = f"❌ Only {successful_tasks}/{total_tasks} analyses completed"
        
        logger.info("\n" + "=" * 60)
        logger.info(f"ANALYSIS COMPLETE: {results['summary']['status']}")
        logger.info("=" * 60 + "\n")
        
        return results
    
    except Exception as e:
        logger.error(f"FATAL Analysis Engine Error: {str(e)}")
        results["has_errors"] = True
        results["summary"]["status"] = f"❌ Engine error: {str(e)[:60]}"
        return results


# Legacy function for backward compatibility
def run_ml_task(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Legacy wrapper around run_regression for backward compatibility.
    """
    return run_regression(df, target_col)
        raise
