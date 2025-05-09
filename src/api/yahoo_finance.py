"""
Yahoo Finance API client module.
This module handles interactions with the yfinance package to fetch financial data.
"""

import logging
import time
from typing import List, Dict, Union, Optional
import pandas as pd
import yfinance as yf
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class YahooFinanceClient:
    """Client for interacting with Yahoo Finance API through yfinance package."""
    
    def __init__(self, rate_limit_pause: float = 0.5):
        """
        Initialize Yahoo Finance client with rate limiting.
        
        Args:
            rate_limit_pause: Time in seconds to pause between API calls
        """
        self.rate_limit_pause = rate_limit_pause
        logger.info("Yahoo Finance client initialized")
    
    def get_stock_data(
        self, 
        ticker: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Time interval between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with stock data or None if request failed
        """
        logger.info(f"Fetching {ticker} data for period {period} at {interval} intervals")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Reset index to make Date a column and rename columns to standardized format
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add ticker column for identification in combined datasets
            data['ticker'] = ticker
            
            logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_multiple_stocks_data(
        self, 
        tickers: List[str], 
        period: str = "1y", 
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period to fetch
            interval: Time interval between data points
        
        Returns:
            Dictionary mapping ticker symbols to their respective DataFrames
        """
        logger.info(f"Fetching data for {len(tickers)} stocks")
        results = {}
        
        for ticker in tickers:
            data = self.get_stock_data(ticker, period, interval)
            if data is not None:
                results[ticker] = data
        
        logger.info(f"Successfully fetched data for {len(results)} out of {len(tickers)} stocks")
        return results
    
    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """
        Get company information for a ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with company information or None if request failed
        """
        logger.info(f"Fetching company information for {ticker}")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if not info:
                logger.warning(f"No information returned for {ticker}")
                return None
                
            logger.info(f"Successfully fetched information for {ticker}")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return info
            
        except Exception as e:
            logger.error(f"Error fetching information for {ticker}: {str(e)}")
            return None
    
    def get_market_news(self) -> Optional[List[Dict]]:
        """
        Get recent market news.
        
        Returns:
            List of news articles or None if request failed
        """
        # yfinance doesn't have a dedicated news API, so we use the news from a market index
        logger.info("Fetching market news")
        
        try:
            # S&P 500 news as proxy for market news
            sp500 = yf.Ticker("^GSPC")
            news = sp500.news
            
            if not news:
                logger.warning("No market news returned")
                return None
                
            logger.info(f"Successfully fetched {len(news)} market news articles")
            time.sleep(self.rate_limit_pause)  # Respect rate limits
            return news
            
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return None