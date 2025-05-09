"""
Financial data analysis functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import pandas_ta as ta
from src.utils.logger import setup_logger
from config.settings import MOVING_AVERAGE_PERIODS, RSI_PERIOD, BOLLINGER_PERIOD

logger = setup_logger(__name__)

def calculate_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 20, 60, 250]) -> pd.DataFrame:
    """
    Calculate returns over different periods.
    
    Args:
        df: DataFrame with stock price data (must have 'close' and 'date' columns)
        periods: List of periods (in trading days) to calculate returns for
        
    Returns:
        DataFrame with additional return columns
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for return calculation")
        return pd.DataFrame()
    
    if 'close' not in df.columns:
        logger.error("DataFrame must have 'close' column to calculate returns")
        return df
    
    logger.info(f"Calculating returns for {len(periods)} periods")
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by date to ensure correct calculation
    if 'date' in result.columns:
        result = result.sort_values('date')
    
    # Calculate returns for each period
    for period in periods:
        col_name = f'return_{period}d'
        result[col_name] = result['close'].pct_change(period) * 100
        
        # Calculate annualized return
        if period > 1:
            annual_factor = 252 / period  # 252 trading days in a year
            annual_col_name = f'return_{period}d_annual'
            result[annual_col_name] = ((1 + result[col_name]/100) ** annual_factor - 1) * 100
    
    logger.info("Return calculation complete")
    return result

def calculate_volatility(df: pd.DataFrame, windows: List[int] = [20, 60, 250]) -> pd.DataFrame:
    """
    Calculate rolling volatility over different windows.
    
    Args:
        df: DataFrame with stock price data (must have 'close' column)
        windows: List of periods to calculate volatility for
        
    Returns:
        DataFrame with additional volatility columns
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for volatility calculation")
        return pd.DataFrame()
    
    if 'close' not in df.columns:
        logger.error("DataFrame must have 'close' column to calculate volatility")
        return df
    
    logger.info(f"Calculating volatility for {len(windows)} windows")
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by date to ensure correct calculation
    if 'date' in result.columns:
        result = result.sort_values('date')
    
    # Calculate daily returns for volatility calculation
    result['daily_return'] = result['close'].pct_change()
    
    # Calculate rolling volatility (annualized)
    for window in windows:
        col_name = f'volatility_{window}d'
        result[col_name] = result['daily_return'].rolling(window=window).std() * np.sqrt(252) * 100
    
    # Drop the temporary column
    result = result.drop(columns=['daily_return'])
    
    logger.info("Volatility calculation complete")
    return result

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate common technical indicators.
    
    Args:
        df: DataFrame with stock price data (must have OHLCV columns)
        
    Returns:
        DataFrame with additional technical indicator columns
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for technical indicator calculation")
        return pd.DataFrame()
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return df
    
    logger.info("Calculating technical indicators")
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Sort by date to ensure correct calculation
    if 'date' in result.columns:
        result = result.sort_values('date')
    
    # Calculate Moving Averages
    for period in MOVING_AVERAGE_PERIODS:
        result[f'sma_{period}'] = ta.sma(result['close'], length=period)
        result[f'ema_{period}'] = ta.ema(result['close'], length=period)
    
    # Calculate RSI
    result[f'rsi_{RSI_PERIOD}'] = ta.rsi(result['close'], length=RSI_PERIOD)
    
    # Calculate MACD
    macd = ta.macd(result['close'])
    # Rename columns to simpler names
    macd.columns = ['macd', 'macd_signal', 'macd_hist']
    result = pd.concat([result, macd], axis=1)
    
    # Calculate Bollinger Bands
    bbands = ta.bbands(result['close'], length=BOLLINGER_PERIOD)
    # Rename columns to simpler names
    bbands.columns = ['bb_lower', 'bb_middle', 'bb_upper']
    result = pd.concat([result, bbands], axis=1)
    
    # Calculate Average True Range (ATR)
    result['atr_14'] = ta.atr(result['high'], result['low'], result['close'], length=14)
    
    # Calculate On-Balance Volume (OBV)
    result['obv'] = ta.obv(result['close'], result['volume'])
    
    logger.info("Technical indicator calculation complete")
    return result

def calculate_financial_ratios(
    income_statement: pd.DataFrame, 
    balance_sheet: pd.DataFrame, 
    stock_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate key financial ratios from financial statements.
    
    Args:
        income_statement: Income statement data
        balance_sheet: Balance sheet data
        stock_data: Recent stock price data
        
    Returns:
        Dictionary of financial ratios
    """
    if income_statement is None or balance_sheet is None or stock_data is None:
        logger.warning("Incomplete data provided for ratio calculation")
        return {}
    
    logger.info("Calculating financial ratios")
    ratios = {}
    
    try:
        # Extract latest data
        latest_income = income_statement.iloc[:, 0]  # Most recent column
        latest_balance = balance_sheet.iloc[:, 0]    # Most recent column
        latest_price = stock_data['close'].iloc[-1]  # Most recent closing price
        
        # Basic financial data
        revenue = float(latest_income.get('totalRevenue', 0))
        net_income = float(latest_income.get('netIncome', 0))
        total_assets = float(latest_balance.get('totalAssets', 0))
        total_liabilities = float(latest_balance.get('totalLiabilities', 0))
        total_equity = float(latest_balance.get('totalShareholderEquity', 0))
        shares_outstanding = float(latest_balance.get('commonStock', 0))
        
        # Calculate ratios
        
        # Profitability ratios
        ratios['gross_margin'] = float(latest_income.get('grossProfit', 0)) / revenue if revenue else None
        ratios['operating_margin'] = float(latest_income.get('operatingIncome', 0)) / revenue if revenue else None
        ratios['net_margin'] = net_income / revenue if revenue else None
        ratios['roe'] = net_income / total_equity if total_equity else None
        ratios['roa'] = net_income / total_assets if total_assets else None
        
        # Liquidity ratios
        current_assets = float(latest_balance.get('totalCurrentAssets', 0))
        current_liabilities = float(latest_balance.get('totalCurrentLiabilities', 0))
        ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities else None
        
        inventory = float(latest_balance.get('inventory', 0))
        ratios['quick_ratio'] = (current_assets - inventory) / current_liabilities if current_liabilities else None
        
        # Efficiency ratios
        ratios['asset_turnover'] = revenue / total_assets if total_assets else None
        
        # Leverage ratios
        ratios['debt_to_equity'] = total_liabilities / total_equity if total_equity else None
        ratios['debt_to_assets'] = total_liabilities / total_assets if total_assets else None
        
        # Market ratios
        ratios['eps'] = net_income / shares_outstanding if shares_outstanding else None
        ratios['pe_ratio'] = latest_price / ratios['eps'] if ratios['eps'] else None
        ratios['price_to_book'] = latest_price / (total_equity / shares_outstanding) if (total_equity and shares_outstanding) else None
        
        # Remove None values
        ratios = {k: v for k, v in ratios.items() if v is not None}
        
        logger.info(f"Calculated {len(ratios)} financial ratios")
        return ratios
        
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {str(e)}")
        return {}

def calculate_portfolio_metrics(
    portfolio_data: Dict[str, pd.DataFrame],
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        portfolio_data: Dictionary mapping tickers to their respective DataFrames
        weights: Dictionary mapping tickers to their weights in the portfolio
                (defaults to equal weighting if not provided)
        
    Returns:
        Dictionary of portfolio metrics
    """
    if not portfolio_data:
        logger.warning("No portfolio data provided")
        return {}
    
    logger.info("Calculating portfolio metrics")
    
    # Create equal weights if not provided
    if weights is None:
        total_stocks = len(portfolio_data)
        weights = {ticker: 1.0 / total_stocks for ticker in portfolio_data.keys()}
    
    # Normalize weights to ensure they sum to 1
    weight_sum = sum(weights.values())
    weights = {ticker: weight / weight_sum for ticker, weight in weights.items()}
    
    try:
        # Calculate daily returns for each stock
        returns_dfs = []
        
        for ticker, df in portfolio_data.items():
            if df is None or df.empty or 'close' not in df.columns:
                continue
                
            # Calculate daily returns
            returns = df.copy()
            returns = returns.sort_values('date')
            returns['daily_return'] = returns['close'].pct_change()
            
            # Add weight column
            returns['weight'] = weights.get(ticker, 0)
            
            # Calculate weighted return
            returns['weighted_return'] = returns['daily_return'] * returns['weight']
            
            # Select only necessary columns
            returns = returns[['date', 'ticker', 'daily_return', 'weight', 'weighted_return']]
            
            returns_dfs.append(returns)
        
        if not returns_dfs:
            logger.warning("No valid return data found")
            return {}
        
        # Combine all returns
        all_returns = pd.concat(returns_dfs)
        
        # Calculate portfolio daily returns
        portfolio_returns = all_returns.groupby('date')['weighted_return'].sum().reset_index()
        portfolio_returns.columns = ['date', 'portfolio_return']
        
        # Calculate metrics
        metrics = {}
        
        # Annual return (252 trading days in a year)
        daily_mean_return = portfolio_returns['portfolio_return'].mean()
        metrics['annual_return'] = ((1 + daily_mean_return) ** 252 - 1) * 100
        
        # Volatility (annualized)
        daily_std = portfolio_returns['portfolio_return'].std()
        metrics['annual_volatility'] = daily_std * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming risk-free rate of 0% for simplicity)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility'] if metrics['annual_volatility'] != 0 else 0
        
        # Maximum drawdown
        portfolio_returns['cumulative_return'] = (1 + portfolio_returns['portfolio_return']).cumprod()
        running_max = portfolio_returns['cumulative_return'].cummax()
        drawdown = (portfolio_returns['cumulative_return'] / running_max - 1) * 100
        metrics['max_drawdown'] = drawdown.min()
        
        logger.info("Portfolio metrics calculation complete")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}