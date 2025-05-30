"""
Main entry point for the Financial API project.
Provides a command-line interface to interact with financial data APIs.
"""

import argparse
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from src.api.alpha_vantage import AlphaVantageClient
from src.data_processing.cleaning import clean_stock_data
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def save_data(df: pd.DataFrame, symbol: str, data_type: str) -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: DataFrame to save
        symbol: Stock symbol
        data_type: Type of data (e.g., 'daily', 'income_statement')
    """
    if df is None or df.empty:
        logger.warning(f"No data to save for {symbol} {data_type}")
        return
        
    # Create data directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'data/processed/{symbol}_{data_type}_{timestamp}.csv'
    
    df.to_csv(filename, index=False)
    logger.info(f"Data saved to {filename}")

def fetch_stock_data(symbols: List[str], data_type: str = 'daily') -> None:
    """
    Fetch and process stock data for given symbols.
    
    Args:
        symbols: List of stock symbols
        data_type: Type of data to fetch ('daily', 'income_statement', 'balance_sheet', 'cash_flow', 'rsi')
    """
    client = AlphaVantageClient()
    total_symbols = len(symbols)
    
    for idx, symbol in enumerate(symbols, 1):
        logger.info(f"Processing {symbol} ({idx}/{total_symbols})...")
        
        try:
            if data_type == 'daily':
                data = client.get_daily_adjusted(symbol)
            elif data_type == 'income_statement':
                data = client.get_income_statement(symbol)
            elif data_type == 'balance_sheet':
                data = client.get_balance_sheet(symbol)
            elif data_type == 'cash_flow':
                data = client.get_cash_flow(symbol)
            elif data_type == 'rsi':
                data = client.get_rsi(symbol)
            else:
                logger.error(f"Unsupported data type: {data_type}")
                continue
            
            if data is not None:
                # Clean the data if it's price data
                if data_type in ['daily', 'rsi']:
                    data = clean_stock_data(data)
                
                # Save the processed data
                save_data(data, symbol, data_type)
            else:
                logger.warning(f"Skipping {symbol} due to API limits or errors")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Financial Data API Client')
    
    parser.add_argument(
        'symbols',
        nargs='+',
        help='Stock symbols to process (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '--data-type',
        choices=['daily', 'income_statement', 'balance_sheet', 'cash_flow', 'rsi'],
        default='daily',
        help='Type of data to fetch (default: daily)'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting data fetch for {len(args.symbols)} symbols")
    logger.info(f"Data type: {args.data_type}")
    logger.info("Note: Free tier has a limit of 5 API calls per minute and 500 calls per day")
    
    fetch_stock_data(args.symbols, args.data_type)
    
    logger.info("Data processing completed")

if __name__ == '__main__':
    main() 