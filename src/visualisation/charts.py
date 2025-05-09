"""
Chart generation functions for financial data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from src.utils.logger import setup_logger
from src.utils.helpers import format_currency

logger = setup_logger(__name__)

# Set styles for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def create_stock_price_chart(df: pd.DataFrame, ticker: str = None) -> go.Figure:
    """
    Create an interactive stock price chart with volume.
    
    Args:
        df: DataFrame with stock price data
        ticker: Stock ticker symbol (optional, will be extracted from df if available)
        
    Returns:
        Plotly figure object
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for chart creation")
        return go.Figure()
    
    required_columns = ['date', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns: {[col for col in required_columns if col not in df.columns]}")
        return go.Figure()
    
    # Extract ticker from DataFrame if not provided
    if ticker is None and 'ticker' in df.columns:
        ticker = df['ticker'].iloc[0]
    
    # Ensure date is in proper format
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    logger.info(f"Creating price chart for {'ticker ' + ticker if ticker else 'unnamed stock'}")
    
    # Create subplot with shared x-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume chart
    if 'volume' in df.columns:
        # Color volume bars based on price direction
        colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Add moving averages if available
    for period in [20, 50, 200]:
        col_name = f'sma_{period}'
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df[col_name],
                    name=f"SMA {period}",
                    line=dict(width=1)
                ),
                row=1, col=1
            )
    
    # Update layout
    title = f"Stock Price Chart: {ticker}" if ticker else "Stock Price Chart"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_technical_indicators_chart(df: pd.DataFrame, ticker: str = None) -> go.Figure:
    """
    Create chart with technical indicators.
    
    Args:
        df: DataFrame with stock price and technical indicator data
        ticker: Stock ticker symbol (optional)
        
    Returns:
        Plotly figure object
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided for technical chart creation")
        return go.Figure()
    
    # Check for required columns
    if 'date' not in df.columns or 'close' not in df.columns:
        logger.error("DataFrame must have at least 'date' and 'close' columns")
        return go.Figure()
    
    # Extract ticker from DataFrame if not provided
    if ticker is None and 'ticker' in df.columns:
        ticker = df['ticker'].iloc[0]
    
    # Ensure date is in proper format
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    logger.info(f"Creating technical indicators chart for {'ticker ' + ticker if ticker else 'unnamed stock'}")
    
    # Create subplot grid
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.5, 0.25, 0.25])
    
    # Add price and Bollinger Bands to first subplot
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close'],
            name="Close Price",
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands if available
    bb_columns = ['bb_lower', 'bb_middle', 'bb_upper']
    if all(col in df.columns for col in bb_columns):
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['bb_upper'],
                name="Upper BB",
                line=dict(color='rgba(250, 120, 120, 0.5)')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['bb_middle'],
                name="Middle BB",
                line=dict(color='rgba(120, 120, 250, 0.5)')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['bb_lower'],
                name="Lower BB",
                line=dict(color='rgba(120, 250, 120, 0.5)')
            ),
            row=1, col=1
        )
    
    # Add RSI to second subplot if available
    rsi_column = 'rsi_14'
    if rsi_column in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[rsi_column],
                name="RSI (14)",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Add RSI reference lines
        fig.add_shape(
            type="line",
            x0=df['date'].min(),
            x1=df['date'].max(),
            y0=70,
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=2, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df['date'].min(),
            x1=df['date'].max(),
            y0=30,
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=2, col=1
        )
    
    # Add MACD to third subplot if available
    macd_columns = ['macd', 'macd_signal', 'macd_hist']
    if all(col in df.columns for col in macd_columns):
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['macd'],
                name="MACD",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['macd_signal'],
                name="Signal Line",
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # Add MACD histogram
        colors = ['red' if val < 0 else 'green' for val in df['macd_hist']]
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['macd_hist'],
                name="MACD Histogram",
                marker_color=colors,
                opacity=0.7
            ),
            row=3, col=1
        )
    
    # Update layout
    title = f"Technical Indicators: {ticker}" if ticker else "Technical Indicators"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        height=800,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def create_comparison_chart(
    dfs: Dict[str, pd.DataFrame], 
    metric: str = 'close',
    normalize: bool = True
) -> go.Figure:
    """
    Create a comparison chart for multiple stocks.
    
    Args:
        dfs: Dictionary mapping tickers to their respective DataFrames
        metric: Column to compare
        normalize: Whether to normalize values to 100 at the start
        
    Returns:
        Plotly figure object
    """
    if not dfs:
        logger.warning("No data provided for comparison chart")
        return go.Figure()
    
    logger.info(f"Creating comparison chart for {len(dfs)} stocks, metric: {metric}")
    
    fig = go.Figure()
    
    for ticker, df in dfs.items():
        if df is None or df.empty or metric not in df.columns or 'date' not in df.columns:
            logger.warning(f"Skipping {ticker}: missing required data")
            continue
        
        # Sort data by date
        df = df.sort_values('date')
        
        # Normalize data if requested
        if normalize:
            first_value = df[metric].iloc[0]
            values = (df[metric] / first_value) * 100
        else:
            values = df[metric]
        
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=values,
                name=ticker,
                mode='lines'
            )
        )
    
    # Update layout
    y_title = "Normalized Value (Start=100)" if normalize else metric.capitalize()
    fig.update_layout(
        title=f"{'Normalized ' if normalize else ''}Comparison: {metric.capitalize()}",
        xaxis_title="Date",
        yaxis_title=y_title,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    
    return fig

def create_correlation_heatmap(dfs: Dict[str, pd.DataFrame], metric: str = 'close') -> go.Figure:
    """
    Create a correlation heatmap for multiple stocks.
    
    Args:
        dfs: Dictionary mapping tickers to their respective DataFrames
        metric: Column to calculate correlations for
        
    Returns:
        Plotly figure object
    """
    if not dfs:
        logger.warning("No data provided for correlation heatmap")
        return go.Figure()
    
    logger.info(f"Creating correlation heatmap for {len(dfs)} stocks")
    
    # Prepare data
    combined_data = pd.DataFrame()
    
    for ticker, df in dfs.items():
        if df is None or df.empty or metric not in df.columns or 'date' not in df.columns:
            logger.warning(f"Skipping {ticker}: missing required data")
            continue
        
        # Extract just the date and metric columns
        subset = df[['date', metric]].copy()
        subset = subset.rename(columns={metric: ticker})
        
        if combined_data.empty:
            combined_data = subset
        else:
            combined_data = pd.merge(combined_data, subset, on='date', how='outer')
    
    if len(combined_data.columns) <= 1:  # Only date column exists
        logger.warning("Insufficient data for correlation heatmap")
        return go.Figure()
    
    # Calculate correlations
    corr_matrix = combined_data.drop(columns=['date']).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation")
    ))
    
    # Add text annotations
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.index[i],
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(
                        color="white" if abs(value) > 0.5 else "black"
                    )
                )
            )
    
    fig.update_layout(
        title="Correlation Matrix",
        height=600,
        width=700,
        annotations=annotations
    )
    
    return fig

def create_financial_ratios_chart(ratios: Dict[str, float]) -> go.Figure:
    """
    Create a bar chart for financial ratios.
    
    Args:
        ratios: Dictionary of financial ratios
        
    Returns:
        Plotly figure object
    """
    if not ratios:
        logger.warning("No financial ratios provided")
        return go.Figure()
    
    logger.info(f"Creating financial ratios chart with {len(ratios)} ratios")
    
    # Categorize ratios
    categories = {
        'Profitability': ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa'],
        'Liquidity': ['current_ratio', 'quick_ratio'],
        'Leverage': ['debt_to_equity', 'debt_to_assets'],
        'Valuation': ['pe_ratio', 'price_to_book'],
        'Other': []
    }
    
    # Create figure with subplots
    fig = make_subplots(rows=len(categories), cols=1, 
                        subplot_titles=list(categories.keys()),
                        vertical_spacing=0.1)
    
    # Plot ratios by category
    for i, (category, ratio_keys) in enumerate(categories.items(), start=1):
        # Filter ratios that belong to this category
        category_ratios = {k: v for k, v in ratios.items() if k in ratio_keys}
        
        # Add any ratios that don't match predefined categories to 'Other'
        if category == 'Other':
            predefined_keys = [key for cat_keys in list(categories.values())[:-1] for key in cat_keys]
            category_ratios.update({k: v for k, v in ratios.items() if k not in predefined_keys})
        
        if not category_ratios:
            continue
        
        # Sort by value
        sorted_ratios = sorted(category_ratios.items(), key=lambda x: x[1])
        labels = [k.replace('_', ' ').title() for k, _ in sorted_ratios]
        values = [v for _, v in sorted_ratios]
        
        # Create bar chart
        