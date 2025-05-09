"""
Alpha Vantage API client module.
This module handles interactions with the Alpha Vantage API to fetch financial data.
"""

import logging
import time
from typing import Dict, Optional, List, Any
import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators
from config.settings import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_CALLS_PER_MINUTE
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AlphaVantageClient:
    """Client for interacting with Alpha Vantage APIs."""
    
    def __init__(self, api_key: str = ALPHA_VANTAGE_API_KEY):
        """
        Initialize Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key
        """
        if not api_key:
            logger.error("Alpha Vantage API key is missing")
            raise ValueError("Alpha Vantage API key is required")
            
        self.api_key = api_key
        self.time_series = TimeSeries(key=api_key, output_format='pandas')
        self.fundamental_data = FundamentalData(key=api_key, output_format='pandas')
        self.tech_indicators = TechIndicators(key=api_key, output_format='pandas')
        
        # Calculate pause time based on rate limits
        self.rate_limit_pause = 60.0 / ALPHA_VANTAGE_CALLS_PER_MINUTE
        
        logger.info("Alpha Vantage client initialized")
    
    def get_daily_adjusted(self, symbol: str, outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """
        Get daily adjusted time series data for a given symbol.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' (last 100 data points) or 'full' (up to 20 years of data)
        
        Returns:
            DataFrame with daily adjusted data or None if request failed
        """
        logger.info(f"Fetching daily adjusted data for {symbol}")
        
        try:
            data, meta_data = self.time_series.get_daily_adjusted(
                symbol=symbol, 
                outputsize=outputsize
            )
            
            # Add ticker column and ensure date is a column, not index
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data['ticker'] = symbol
            
            logger.info(f"Successfully fetched {len(data)} days of data for {symbol}")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return data
        
        except Exception as e:
            logger.error(f"Error fetching daily adjusted data for {symbol}: {str(e)}")
            return None
    
    def get_income_statement(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get annual income statement for a company.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with income statement data or None if request failed
        """
        logger.info(f"Fetching income statement for {symbol}")
        
        try:
            data, meta_data = self.fundamental_data.get_income_statement_annual(symbol=symbol)
            
            if data.empty:
                logger.warning(f"No income statement data returned for {symbol}")
                return None
                
            logger.info(f"Successfully fetched income statement for {symbol}")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return data
        
        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {str(e)}")
            return None
    
    def get_balance_sheet(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get annual balance sheet for a company.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with balance sheet data or None if request failed
        """
        logger.info(f"Fetching balance sheet for {symbol}")
        
        try:
            data, meta_data = self.fundamental_data.get_balance_sheet_annual(symbol=symbol)
            
            if data.empty:
                logger.warning(f"No balance sheet data returned for {symbol}")
                return None
                
            logger.info(f"Successfully fetched balance sheet for {symbol}")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return data
        
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {str(e)}")
            return None
    
    def get_cash_flow(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get annual cash flow statement for a company.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with cash flow data or None if request failed
        """
        logger.info(f"Fetching cash flow statement for {symbol}")
        
        try:
            data, meta_data = self.fundamental_data.get_cash_flow_annual(symbol=symbol)
            
            if data.empty:
                logger.warning(f"No cash flow data returned for {symbol}")
                return None
                
            logger.info(f"Successfully fetched cash flow statement for {symbol}")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return data
        
        except Exception as e:
            logger.error(f"Error fetching cash flow statement for {symbol}: {str(e)}")
            return None
    
    def get_rsi(self, symbol: str, interval: str = 'daily', time_period: int = 14) -> Optional[pd.DataFrame]:
        """
        Get Relative Strength Index (RSI) for a stock.
        
        Args:
            symbol: Stock symbol
            interval: Time interval between data points
            time_period: Number of data points used to calculate RSI
        
        Returns:
            DataFrame with RSI data or None if request failed
        """
        logger.info(f"Fetching RSI for {symbol} with period {time_period}")
        
        try:
            data, meta_data = self.tech_indicators.get_rsi(
                symbol=symbol,
                interval=interval,
                time_period=time_period
            )
            
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data['ticker'] = symbol
            
            logger.info(f"Successfully fetched RSI data for {symbol}")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return data
        
        except Exception as e:
            logger.error(f"Error fetching RSI data for {symbol}: {str(e)}")
            return None