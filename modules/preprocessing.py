# modules/preprocessing.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from logger import get_logger
from config import OUTLIER_Z_THRESHOLD
from datetime import datetime

logger = get_logger("preprocessing")


def safe_mode(series: pd.Series) -> Any:
    """
    Safely calculate the mode of a series, handling all edge cases.
    Never crashes, always returns a value.
    
    Args:
        series: pd.Series to calculate mode for
        
    Returns:
        The mode value as a string, or default fallback
    """
    try:
        if series.isnull().all():
            return "Unknown"
        
        # Remove nulls
        non_null = series.dropna()
        if len(non_null) == 0:
            return "Unknown"
        
        # Try to get mode
        mode_result = non_null.mode()
        if len(mode_result) > 0:
            return str(mode_result.iloc[0])
        else:
            return str(non_null.iloc[0])  # Return first non-null value
    except Exception as e:
        logger.warn(f"Mode calculation failed: {str(e)}. Using 'Unknown'")
        return "Unknown"


def is_numeric_convertible(val: Any) -> bool:
    """Check if a value can be converted to numeric."""
    if pd.isna(val):
        return True
    if isinstance(val, (int, float)):
        return True
    
    # Check string values
    str_val = str(val).strip().lower()
    if str_val in ['nan', 'null', 'none', 'n/a', 'na', '', 'unknown']:
        return False
    
    try:
        float(str_val)
        return True
    except (ValueError, TypeError):
        return False


def convert_to_numeric_safe(val: Any, col_name: str = "") -> Any:
    """
    Safely convert a value to numeric, returning NaN on failure.
    """
    if pd.isna(val):
        return np.nan
    
    if isinstance(val, (int, float)):
        return float(val)
    
    str_val = str(val).strip().lower()
    if str_val in ['nan', 'null', 'none', 'n/a', 'na', '', 'unknown']:
        return np.nan
    
    try:
        return float(str_val)
    except (ValueError, TypeError):
        logger.debug(f"Could not convert '{val}' to numeric in column {col_name}")
        return np.nan


def try_parse_date(date_str: Any) -> Any:
    """
    Attempt to parse various date formats safely.
    Returns NaT on failure.
    """
    if pd.isna(date_str):
        return pd.NaT
    
    if isinstance(date_str, (pd.Timestamp, datetime)):
        return pd.Timestamp(date_str)
    
    str_val = str(date_str).strip()
    
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%m-%d-%Y',
        '%Y/%m/%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%B %d, %Y',
        '%d %B %Y',
        '%b %d, %Y',
        '%d %b %Y',
    ]
    
    for fmt in date_formats:
        try:
            return pd.Timestamp(datetime.strptime(str_val, fmt))
        except (ValueError, TypeError):
            continue
    
    # Try pandas default parser as last resort
    try:
        return pd.Timestamp(str_val)
    except (ValueError, TypeError):
        return pd.NaT


def safe_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    SAFE PREPROCESSING: Handles ANY level of data corruption without crashing.
    
    Never fails. Always returns a cleaned DataFrame and detailed audit report.
    
    Handles:
    1. Missing values (NaN, null, empty cells)
    2. String values in numeric columns ("abc", "N/A", "unknown")
    3. Negative values where impossible (age = -5)
    4. Extreme outliers (age = 999, revenue = 999999999)
    5. Duplicate rows
    6. Invalid dates (13-16-2023, "yesterday", etc.)
    7. Completely empty rows or columns
    8. Mixed data types in same column
    9. mode() operation on string/object columns (SAFE MODE)
    
    Args:
        df: Input DataFrame (can be corrupted)
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (Cleaned DataFrame, Detailed Audit Report)
    """
    
    audit = {
        "total_rows_input": len(df),
        "total_columns_input": len(df.columns),
        "empty_rows_removed": 0,
        "duplicate_rows_removed": 0,
        "completely_empty_columns_removed": [],
        "missing_values_handled": {},
        "string_values_converted": {},
        "negative_values_corrected": {},
        "outliers_detected": {},
        "invalid_dates_converted": {},
        "mixed_type_columns": [],
        "errors": [],
    }
    
    try:
        # Start with a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # ======== STEP 1: Handle completely empty dataframe ========
        if cleaned_df.empty or len(cleaned_df) == 0:
            logger.warn("Input DataFrame is empty")
            audit["errors"].append("Input DataFrame is empty")
            return pd.DataFrame(), audit
        
        # ======== STEP 2: Remove completely empty rows ========
        before_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(how='all')
        empty_rows_removed = before_rows - len(cleaned_df)
        if empty_rows_removed > 0:
            audit["empty_rows_removed"] = empty_rows_removed
            logger.info(f"Removed {empty_rows_removed} completely empty rows")
        
        # ======== STEP 3: Remove duplicate rows ========
        before_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = before_rows - len(cleaned_df)
        if duplicates_removed > 0:
            audit["duplicate_rows_removed"] = duplicates_removed
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # ======== STEP 4: Remove completely empty columns ========
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().all():
                audit["completely_empty_columns_removed"].append(col)
                logger.warn(f"Removing completely empty column: {col}")
        cleaned_df = cleaned_df.dropna(axis=1, how='all')
        
        # ======== STEP 5: Process each column ========
        for col in cleaned_df.columns:
            try:
                # Detect column type expectations based on name
                col_lower = col.lower()
                is_age_col = 'age' in col_lower
                is_numeric_col = any(x in col_lower for x in ['age', 'rating', 'reviews', 'discount', 'quantity', 'price', 'days', 'revenue', 'num_'])
                is_date_col = any(x in col_lower for x in ['date', 'time'])
                is_categorical_col = any(x in col_lower for x in ['gender', 'category', 'region', 'payment', 'method', 'returned'])
                
                # Convert to proper type if numeric is expected
                if is_numeric_col and cleaned_df[col].dtype == 'object':
                    # Column has string values but should be numeric
                    non_numeric_count = 0
                    converted_values = []
                    
                    for idx, val in cleaned_df[col].items():
                        if not is_numeric_convertible(val):
                            non_numeric_count += 1
                            converted_values.append(np.nan)
                        else:
                            converted_values.append(convert_to_numeric_safe(val, col))
                    
                    if non_numeric_count > 0:
                        audit["string_values_converted"][col] = non_numeric_count
                        logger.info(f"Column '{col}': Converted {non_numeric_count} non-numeric values to NaN")
                    
                    cleaned_df[col] = pd.to_numeric(converted_values, errors='coerce')
                
                # Convert to datetime if date column is expected
                elif is_date_col and cleaned_df[col].dtype == 'object':
                    invalid_dates = 0
                    converted_dates = []
                    
                    for val in cleaned_df[col]:
                        parsed = try_parse_date(val)
                        if pd.isna(parsed):
                            invalid_dates += 1
                        converted_dates.append(parsed)
                    
                    if invalid_dates > 0:
                        audit["invalid_dates_converted"][col] = invalid_dates
                        logger.info(f"Column '{col}': {invalid_dates} invalid dates converted to NaT")
                    
                    cleaned_df[col] = converted_dates
                
                # ======== STEP 6: Handle negative/impossible values ========
                if is_numeric_col and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    if is_age_col:
                        # Age should not be negative or > 150
                        invalid_mask = (cleaned_df[col] < 0) | (cleaned_df[col] > 150)
                        invalid_count = invalid_mask.sum()
                        if invalid_count > 0:
                            audit["negative_values_corrected"][col] = int(invalid_count)
                            cleaned_df.loc[invalid_mask, col] = np.nan
                            logger.info(f"Column '{col}': Corrected {invalid_count} invalid age values")
                    elif 'rating' in col_lower:
                        # Rating typically 0-5
                        invalid_mask = (cleaned_df[col] < 0) | (cleaned_df[col] > 5)
                        invalid_count = invalid_mask.sum()
                        if invalid_count > 0:
                            audit["negative_values_corrected"][col] = int(invalid_count)
                            cleaned_df.loc[invalid_mask, col] = np.nan
                            logger.info(f"Column '{col}': Corrected {invalid_count} invalid rating values")
                    elif 'quantity' in col_lower or 'discount' in col_lower:
                        # Quantity and discount should not be negative
                        invalid_mask = cleaned_df[col] < 0
                        invalid_count = invalid_mask.sum()
                        if invalid_count > 0:
                            audit["negative_values_corrected"][col] = int(invalid_count)
                            cleaned_df.loc[invalid_mask, col] = np.nan
                            logger.info(f"Column '{col}': Corrected {invalid_count} negative values")
                
                # ======== STEP 7: Handle missing values ========
                null_count = cleaned_df[col].isnull().sum()
                if null_count > 0:
                    audit["missing_values_handled"][col] = int(null_count)
                    
                    if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        # For numeric: use median
                        try:
                            median_val = cleaned_df[col].median()
                            if pd.notna(median_val):
                                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                            else:
                                cleaned_df[col] = cleaned_df[col].fillna(0)
                        except Exception as e:
                            logger.warn(f"Could not calculate median for {col}, using 0")
                            cleaned_df[col] = cleaned_df[col].fillna(0)
                    elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                        # For dates: use most common date or drop
                        try:
                            mode_date = cleaned_df[col].mode()
                            if len(mode_date) > 0:
                                cleaned_df[col] = cleaned_df[col].fillna(mode_date[0])
                            else:
                                cleaned_df[col] = cleaned_df[col].fillna(pd.Timestamp.now())
                        except Exception as e:
                            logger.warn(f"Could not handle dates for {col}, using current date")
                            cleaned_df[col] = cleaned_df[col].fillna(pd.Timestamp.now())
                    else:
                        # For categorical: use safe_mode
                        mode_val = safe_mode(cleaned_df[col])
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    
                    logger.info(f"Column '{col}': Handled {null_count} missing values")
                
                # ======== STEP 8: Handle outliers (Z-score for numeric) ========
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    try:
                        mean = cleaned_df[col].mean()
                        std = cleaned_df[col].std()
                        
                        if std > 0:  # Avoid division by zero
                            z_scores = np.abs((cleaned_df[col] - mean) / std)
                            outliers = z_scores > OUTLIER_Z_THRESHOLD
                            outlier_count = outliers.sum()
                            
                            if outlier_count > 0:
                                audit["outliers_detected"][col] = int(outlier_count)
                                # Winsorize: replace with median
                                median = cleaned_df[col].median()
                                cleaned_df.loc[outliers, col] = median
                                logger.info(f"Column '{col}': Detected and capped {outlier_count} outliers")
                    except Exception as e:
                        logger.warn(f"Could not detect outliers for {col}: {str(e)}")
            
            except Exception as col_error:
                logger.error(f"Error processing column '{col}': {str(col_error)}")
                audit["errors"].append(f"Column '{col}': {str(col_error)}")
        
        # ======== STEP 9: Final data type consistency check ========
        for col in cleaned_df.columns:
            unique_types = set()
            for val in cleaned_df[col]:
                if pd.notna(val):
                    unique_types.add(type(val).__name__)
            
            if len(unique_types) > 1:
                audit["mixed_type_columns"].append(col)
                logger.warn(f"Column '{col}' has mixed types: {unique_types}")
        
        # ======== Final audit summary ========
        audit["total_rows_output"] = len(cleaned_df)
        audit["total_columns_output"] = len(cleaned_df.columns)
        audit["rows_removed_total"] = audit["total_rows_input"] - audit["total_rows_output"]
        audit["columns_removed_total"] = len(audit["completely_empty_columns_removed"])
        
        logger.info(f"Preprocessing complete: {audit['total_rows_output']} rows, "
                   f"{audit['total_columns_output']} columns")
        logger.info(f"Audit: Removed {audit['rows_removed_total']} rows, "
                   f"{audit['columns_removed_total']} columns")
        
        return cleaned_df, audit
    
    except Exception as e:
        logger.error(f"FATAL Preprocessing error: {str(e)}")
        audit["errors"].append(f"FATAL: {str(e)}")
        # Return empty dataframe with error info rather than crashing
        return pd.DataFrame(), audit


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Legacy function wrapper around safe_preprocess for backward compatibility.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple[pd.DataFrame, Dict]: (Cleaned DataFrame, Audit Report)
    """
    return safe_preprocess(df)
