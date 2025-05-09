"""
Data cleaning functions for financial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare stock price data.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for cleaning")
        return pd.DataFrame()
    
    logger.info(f"Cleaning stock data with {len(df)} rows")
    
    # Make a copy to avoid modifying the original
    cleaned = df.copy()
    
    # Convert date column to datetime if it exists
    if 'date' in cleaned.columns:
        cleaned['date'] = pd.to_datetime(cleaned['date'])
    
    # Standardize column names (lowercase, underscores)
    cleaned.columns = [col.lower().replace(' ', '_') for col in cleaned.columns]
    
    # Handle missing values
    for col in cleaned.columns:
        # For numeric columns, fill missing values with the previous value
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            missing_count = cleaned[col].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in column {col}")
                cleaned[col] = cleaned[col].fillna(method='ffill')
                # If there are still NAs (e.g., at the beginning), fill with the next value
                cleaned[col] = cleaned[col].fillna(method='bfill')
    
    # Check for duplicate dates
    if 'date' in cleaned.columns:
        duplicates = cleaned.duplicated(subset=['date', 'ticker'], keep='first').sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate entries. Keeping first occurrence.")
            cleaned = cleaned.drop_duplicates(subset=['date', 'ticker'], keep='first')
    
    # Remove outliers (values more than 3 standard deviations from the mean)
    # Only apply to price and volume columns to avoid removing legitimate spikes
    price_columns = [col for col in cleaned.columns if any(x in col for x in ['open', 'high', 'low', 'close', 'price'])]
    
    for col in price_columns:
        if col in cleaned.columns and pd.api.types.is_numeric_dtype(cleaned[col]):
            mean = cleaned[col].mean()
            std = cleaned[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers = ((cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)).sum()
            if outliers > 0:
                logger.warning(f"Found {outliers} outliers in {col}")
                # Instead of removing outliers, winsorize them (cap at the boundaries)
                cleaned[col] = cleaned[col].clip(lower=lower_bound, upper=upper_bound)
    
    logger.info(f"Finished cleaning. Final shape: {cleaned.shape}")
    return cleaned