# modules/ingestion.py
import pandas as pd
import os
from typing import Tuple, Optional, Any
from logger import get_logger
from config import MAX_FILE_SIZE_MB, MAX_ROWS_FULL_LOAD, SAMPLE_ROWS, SUPPORTED_FORMATS

logger = get_logger("ingestion")

def validate_file(uploaded_file: Any) -> Tuple[bool, str]:
    """
    Validates the uploaded file for format and size.
    
    Args:
        uploaded_file: The file object from streamlit
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # 1. Check extension (case-insensitive)
        filename = uploaded_file.name
        _, extension = os.path.splitext(filename)
        extension = extension.lower()
        
        if extension not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {extension}")
            return False, f"File format not supported! Please upload {', '.join(SUPPORTED_FORMATS)}"
            
        # 2. Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {file_size_mb:.2f}MB")
            return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit."
            
        return True, ""
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False, f"An error occurred during file validation: {str(e)}"

def load_data(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads data from the uploaded file into a pandas DataFrame.
    Handles chunked loading for large CSV files.
    
    Args:
        uploaded_file: The file object from streamlit
        
    Returns:
        Tuple[Optional[pd.DataFrame], str]: (dataframe, status_message)
    """
    try:
        filename = uploaded_file.name
        _, extension = os.path.splitext(filename)
        extension = extension.lower()
        
        logger.info(f"Loading file: {filename}")
        
        df = None
        
        if extension == ".csv":
            # Check row count for chunked loading
            # We read just the header first to check
            temp_df = pd.read_csv(uploaded_file, nrows=MAX_ROWS_FULL_LOAD + 1)
            uploaded_file.seek(0) # Reset file pointer
            
            if len(temp_df) > MAX_ROWS_FULL_LOAD:
                logger.info(f"Large file detected (> {MAX_ROWS_FULL_LOAD} rows). Using chunked loading.")
                chunks = []
                # Read in chunks and sample
                for chunk in pd.read_csv(uploaded_file, chunksize=SAMPLE_ROWS):
                    chunks.append(chunk)
                    if len(pd.concat(chunks)) >= SAMPLE_ROWS:
                        break
                df = pd.concat(chunks).head(SAMPLE_ROWS)
                logger.info(f"Sampled {len(df)} rows from large CSV.")
            else:
                df = pd.read_csv(uploaded_file)
                
        elif extension == ".xlsx":
            df = pd.read_excel(uploaded_file)
            
        elif extension == ".json":
            df = pd.read_json(uploaded_file)
            
        if df is not None:
            logger.info(f"Successfully loaded {filename} with {len(df)} rows and {len(df.columns)} columns.")
            return df, "Success"
        else:
            return None, "Failed to load data."
            
    except Exception as e:
        logger.error(f"Load error: {str(e)}")
        return None, f"Engine Error: Could not read the file. {str(e)}"
