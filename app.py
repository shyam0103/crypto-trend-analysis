import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import base64
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import io
import threading

# Set page config
st.set_page_config(layout="wide", page_title="Real-Time Trading Dashboard")
st.title("📈 Advanced Analytics of Stock and Cryptocurrency Trends")

# Initialize session state for price tracking
if 'current_price' not in st.session_state:
    st.session_state.current_price = None
if 'prev_price' not in st.session_state:
    st.session_state.prev_price = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'price_history' not in st.session_state:
    st.session_state.price_history = []

# Create tabs for dashboard sections
tab1, tab2, tab3 = st.tabs(["Live Dashboard", "Analytics", "Reports"])

# Sidebar
st.sidebar.header("Settings")
data_type = st.sidebar.selectbox("Asset Type", ["Cryptocurrency", "Stock"])

# Global variables
symbol = None

# Function to get real-time price from Binance
def get_binance_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return float(data['price'])
    except Exception as e:
        st.sidebar.error(f"Error fetching price: {e}")
    return None

# Function to get real-time price from Yahoo Finance
def get_yahoo_price(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return float(data['Close'].iloc[-1])
    except Exception as e:
        st.sidebar.error(f"Error fetching price: {e}")
    return None

# Function to get historical data
def get_historical_data():
    global symbol
    
    if data_type == "Cryptocurrency":
        coin = st.sidebar.selectbox("Select Coin", ["bitcoin", "ethereum", "solana", "ripple", "dogecoin"])
        coin_map = {
            "bitcoin": "BTCUSDT", 
            "ethereum": "ETHUSDT", 
            "solana": "SOLUSDT", 
            "ripple": "XRPUSDT", 
            "dogecoin": "DOGEUSDT"
        }
        symbol = coin_map[coin]
        
        try:
            # Get historical data from Binance
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=90"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data, columns=["ds", "Open", "High", "Low", "Close", "Volume", "CloseTime", 
                                             "QuoteAssetVolume", "NumberOfTrades", "TakerBuyBaseAssetVolume", 
                                             "TakerBuyQuoteAssetVolume", "Ignore"])
            df["ds"] = pd.to_datetime(df["ds"], unit="ms")
            df["y"] = df["Close"].astype(float)
            df["Open"] = df["Open"].astype(float)
            df["High"] = df["High"].astype(float)
            df["Low"] = df["Low"].astype(float)
            df["Volume"] = df["Volume"].astype(float)
            
            # Get real-time price
            current_price = get_binance_price(symbol)
            if not current_price:
                current_price = float(df["y"].iloc[-1])
            
            # Initialize session state with current price
            st.session_state.current_price = current_price
            st.session_state.prev_price = current_price
            
            return df
            
        except Exception as e:
            with tab1:
                st.error(f"❌ Failed to fetch data from Binance API: {str(e)}")
            st.stop()
    else:
        symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
        period = st.sidebar.selectbox("Duration", ["3mo", "6mo", "1y"])
        
        try:
            data = yf.download(symbol, period=period)
            if data.empty:
                with tab1:
                    st.error("❌ No data found for this symbol")
                st.stop()
                
            df = data.reset_index()
            df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
            
            # Get real-time price
            current_price = get_yahoo_price(symbol)
            if not current_price:
                current_price = float(df["y"].iloc[-1])
            
            # Initialize session state with current price
            st.session_state.current_price = current_price
            st.session_state.prev_price = current_price
            
            return df
            
        except Exception as e:
            with tab1:
                st.error(f"❌ Failed to fetch stock data: {str(e)}")
            st.stop()
    
    return None

# Get historical data
df = get_historical_data()

# Forecast settings
days = st.sidebar.slider("Forecast Days", 7, 90, 30)
show_intervals = st.sidebar.checkbox("Show Confidence Intervals", value=True)
update_interval = st.sidebar.slider("Price Update Interval (seconds)", 5, 60, 5)

# Create and train Prophet model
@st.cache_data(ttl=1800)
def create_forecast(df, days):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast, model

# Generate forecast
forecast, model = create_forecast(df, days)

# Function to create figures for PDF report
def create_figure_for_pdf(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format='png', width=800, height=400)
    buf.seek(0)
    return buf

# Function to create PDF report with charts
def create_pdf_report(df, forecast, current_price, symbol, insights, model):
    # Create a PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add title
    elements.append(Paragraph(f"Trading Analysis Report: {symbol}", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Current price and forecast summary
    price_data = [
        ["Current Price", f"${current_price:.2f}"],
        ["Predicted Price (in {0} days)".format(days), f"${forecast['yhat'].iloc[-1]:.2f}"],
        ["Change", f"{((forecast['yhat'].iloc[-1]/current_price)-1)*100:.2f}%"],
        ["Price Range (Last 90 days)", f"${df['y'].min():.2f} - ${df['y'].max():.2f}"],
        ["Average Price", f"${df['y'].mean():.2f}"]
    ]
    
    price_table = Table(price_data, colWidths=[200, 100])
    price_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(Paragraph("Price Summary", heading2_style))
    elements.append(price_table)
    elements.append(Spacer(1, 20))
    
    # Investment insights
    elements.append(Paragraph("Investment Insights", heading2_style))
    elements.append(Paragraph(insights, normal_style))
    elements.append(Spacer(1, 20))
    
    # Create main forecast chart
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df['ds'], y=df['y'], mode='lines', name='Historical Price', line=dict(color='blue')
    ))
    
    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='orange')
    ))
    
    # Confidence intervals
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound',
        line=dict(dash='dot', color='green'), opacity=0.3
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound',
        line=dict(dash='dot', color='red'), opacity=0.3, fill='tonexty'
    ))
    
    fig_forecast.update_layout(
        title="Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=400
    )
    
    # Convert forecast chart to image
    forecast_img_buf = create_figure_for_pdf(fig_forecast)
    forecast_img = Image(forecast_img_buf, width=700, height=300)
    elements.append(Paragraph("Price Forecast Chart", heading2_style))
    elements.append(forecast_img)
    elements.append(Spacer(1, 20))
    
    # Create seasonal components chart using model
    fig_comp = model.plot_components(forecast)
    fig_comp_buf = BytesIO()
    fig_comp.savefig(fig_comp_buf, format='png', bbox_inches='tight')
    fig_comp_buf.seek(0)
    seasonal_img = Image(fig_comp_buf, width=700, height=400)
    elements.append(Paragraph("Seasonal Components", heading2_style))
    elements.append(seasonal_img)
    elements.append(Spacer(1, 20))
    
    # Create price distribution chart
    fig_dist = px.histogram(df, x="y", nbins=30, title="Historical Price Distribution")
    fig_dist.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency", template="plotly_white")
    dist_img_buf = create_figure_for_pdf(fig_dist)
    dist_img = Image(dist_img_buf, width=350, height=300)
    
    # Calculate residuals for error analysis
    historical_forecast = forecast[forecast['ds'] <= df['ds'].max()]
    merged_df = pd.merge(df, historical_forecast, on='ds', how='inner')
    merged_df['error'] = merged_df['y'] - merged_df['yhat']
    merged_df['error_pct'] = (merged_df['error'] / merged_df['y']) * 100
    
    # Create forecast accuracy chart
    fig_error = px.scatter(merged_df, x='y', y='yhat', title="Actual vs Predicted Prices",
                     labels={'y': 'Actual Price', 'yhat': 'Predicted Price'})
    fig_error.add_shape(type="line", line=dict(dash='dash', color='red'),
                  x0=merged_df['y'].min(), y0=merged_df['y'].min(),
                  x1=merged_df['y'].max(), y1=merged_df['y'].max())
    fig_error.update_layout(template="plotly_white")
    error_img_buf = create_figure_for_pdf(fig_error)
    error_img = Image(error_img_buf, width=350, height=300)
    
    # Create error distribution chart
    fig_error_dist = px.histogram(merged_df, x="error_pct", nbins=20, title="Prediction Error Distribution (%)")
    fig_error_dist.update_layout(xaxis_title="Error (%)", yaxis_title="Frequency", template="plotly_white")
    error_dist_buf = create_figure_for_pdf(fig_error_dist)
    error_dist_img = Image(error_dist_buf, width=350, height=300)
    
    # Add analytics charts
    elements.append(Paragraph("Price Analytics", heading2_style))
    
    # Add analytics charts in a side-by-side layout
    data = [
        [dist_img, error_img],
        [error_dist_img, ""]
    ]
    
    # Create table for side-by-side layout
    chart_table = Table(data, colWidths=[350, 350])
    chart_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    elements.append(chart_table)
    elements.append(Spacer(1, 20))
    
    # Create volatility chart
    if 'daily_return' not in df.columns:
        df['daily_return'] = df['y'].pct_change() * 100
    if 'volatility' not in df.columns:
        df['volatility'] = df['daily_return'].rolling(window=7).std()
    
    fig_vol = px.line(df.dropna(), x='ds', y='volatility', title="7-Day Rolling Volatility (%)")
    fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volatility (%)", template="plotly_white")
    vol_img_buf = create_figure_for_pdf(fig_vol)
    vol_img = Image(vol_img_buf, width=700, height=300)
    
    elements.append(Paragraph("Price Volatility", heading2_style))
    elements.append(vol_img)
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Function to update chart with real-time price
def update_chart(df, forecast, price):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'], 
        y=df['y'], 
        mode='lines', 
        name='Historical Price',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        mode='lines', 
        name='Forecast',
        line=dict(color='orange')
    ))
    
    # Confidence intervals
    if show_intervals:
        fig.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat_upper'], 
            mode='lines', 
            name='Upper Bound',
            line=dict(dash='dot', color='green'),
            opacity=0.3
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], 
            y=forecast['yhat_lower'], 
            mode='lines', 
            name='Lower Bound',
            line=dict(dash='dot', color='red'),
            opacity=0.3,
            fill='tonexty'
        ))
    
    # Add recent price history (last 10 points if available)
    price_history = st.session_state.price_history
    if len(price_history) > 1:
        history_times = [p[0] for p in price_history[-10:]]
        history_prices = [p[1] for p in price_history[-10:]]
        
        fig.add_trace(go.Scatter(
            x=history_times,
            y=history_prices,
            mode='lines+markers',
            name='Live Prices',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))
    
    # Current price marker
    current_date = datetime.now()
    fig.add_trace(go.Scatter(
        x=[current_date], 
        y=[price], 
        mode="markers+text",
        text=[f"Now: ${price:.2f}"],
        textposition="top right",
        marker=dict(color='red', size=10),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=500
    )
    
    return fig

# Investment insights based on forecast
def get_insights(forecast, price):
    last_pred = forecast['yhat'].iloc[-1]
    change = ((last_pred / price) - 1) * 100
    
    if change > 5:
        insight = f"📈 Bullish trend - Predicted price: ${last_pred:.2f} (+{change:.1f}%)"
        return insight, "success"
    elif change < -5:
        insight = f"📉 Bearish trend - Predicted price: ${last_pred:.2f} ({change:.1f}%)"
        return insight, "error"
    else:
        insight = f"↔️ Sideways trend - Predicted price: ${last_pred:.2f} ({change:.1f}%)"
        return insight, "warning"

# Set up the analytics tab
with tab2:
    st.subheader("Analytical Insights")
    
    # Create two columns for analytics
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal components
        st.subheader("Seasonal Components")
        fig_comp = model.plot_components(forecast)
        st.pyplot(fig_comp)
        
        # Price distribution
        st.subheader("Price Distribution")
        fig_dist = px.histogram(df, x="y", nbins=30, title="Historical Price Distribution")
        fig_dist.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency")
        st.plotly_chart(fig_dist)
    
    with col2:
        # Prediction error analysis
        st.subheader("Forecast Accuracy")
        
        # Calculate residuals
        historical_forecast = forecast[forecast['ds'] <= df['ds'].max()]
        merged_df = pd.merge(df, historical_forecast, on='ds', how='inner')
        merged_df['error'] = merged_df['y'] - merged_df['yhat']
        merged_df['error_pct'] = (merged_df['error'] / merged_df['y']) * 100
        
        fig_error = px.scatter(merged_df, x='y', y='yhat', 
                           title="Actual vs Predicted Prices",
                           labels={'y': 'Actual Price', 'yhat': 'Predicted Price'})
        fig_error.add_shape(type="line", line=dict(dash='dash', color='red'),
                          x0=merged_df['y'].min(), y0=merged_df['y'].min(),
                          x1=merged_df['y'].max(), y1=merged_df['y'].max())
        st.plotly_chart(fig_error)
        
        # Error distribution
        st.subheader("Prediction Error Distribution")
        fig_error_dist = px.histogram(merged_df, x="error_pct", nbins=20, 
                                   title="Prediction Error Distribution (%)")
        fig_error_dist.update_layout(xaxis_title="Error (%)", yaxis_title="Frequency")
        st.plotly_chart(fig_error_dist)
    
    # Volatility and trends
    st.subheader("Price Volatility & Trends")
    
    # Calculate daily returns
    df['daily_return'] = df['y'].pct_change() * 100
    
    # 7-day rolling volatility
    df['volatility'] = df['daily_return'].rolling(window=7).std()
    
    # Create volatility chart
    fig_vol = px.line(df.dropna(), x='ds', y='volatility', 
                   title="7-Day Rolling Volatility (%)")
    fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volatility (%)")
    st.plotly_chart(fig_vol)

# Set up the reports tab
with tab3:
    st.subheader("Download Trading Analysis Report")
    
    # Generate insights for the report
    report_insight, _ = get_insights(forecast, st.session_state.current_price)
    
    st.write("Generate a comprehensive PDF report with all the analysis and forecasts")
    st.write("The report includes:")
    st.write("- Price summary and forecast")
    st.write("- Seasonal components analysis")
    st.write("- Price distribution and volatility")
    st.write("- Forecast accuracy analysis")
    st.write("- Investment insights")
    
    if st.button("Generate PDF Report"):
        with st.spinner('Generating PDF report...'):
            pdf_buffer = create_pdf_report(df, forecast, st.session_state.current_price, symbol, report_insight, model)
            
            # Create download link
            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="trading_report_{symbol}.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("Report generated successfully! Click the link above to download.")

# Set up the live dashboard tab
with tab1:
    # Price display container
    price_display = st.empty()
    
    # Create main chart
    chart_container = st.container()
    chart_placeholder = chart_container.empty()
    
    # Investment insights
    insight_container = st.container()
    insight_placeholder = insight_container.empty()
    
    # Get initial prices
    updated_price = get_binance_price(symbol) if data_type == "Cryptocurrency" else get_yahoo_price(symbol)
    if not updated_price:
        updated_price = st.session_state.current_price
    
    # Store previous price for comparison
    st.session_state.prev_price = st.session_state.current_price
    st.session_state.current_price = updated_price
    
    # Store price history for chart tracking
    st.session_state.price_history.append((datetime.now(), updated_price))
    if len(st.session_state.price_history) > 100:  # Limit history to 100 points
        st.session_state.price_history.pop(0)
    
    # Calculate price change
    price_change = st.session_state.current_price - st.session_state.prev_price
    price_change_pct = (price_change / st.session_state.prev_price) * 100 if st.session_state.prev_price else 0
    
    # Update price display
    price_display.metric(
        f"💰 Real-Time Price: {symbol}", 
        f"${st.session_state.current_price:.2f}", 
        f"{price_change:.4f} ({price_change_pct:.2f}%)"
    )
    
    # Update chart
    chart = update_chart(df, forecast, st.session_state.current_price)
    chart_placeholder.plotly_chart(chart, use_container_width=True)
    
    # Update insights
    insight_text, insight_type = get_insights(forecast, st.session_state.current_price)
    if insight_type == "success":
        insight_placeholder.success(insight_text)
    elif insight_type == "error":
        insight_placeholder.error(insight_text)
    else:
        insight_placeholder.warning(insight_text)
    
    # Status indicator and timing
    st.session_state.last_update_time = datetime.now()
    st.caption(f"✅ Last updated: {st.session_state.last_update_time.strftime('%H:%M:%S')} | Auto-refreshes every {update_interval} seconds")
    
    # Create price history table
    if st.checkbox("Show Price History"):
        st.subheader("Recent Price Updates")
        recent_history = st.session_state.price_history[-10:]
        history_df = pd.DataFrame(recent_history, columns=["Timestamp", "Price"])
        history_df["Time"] = history_df["Timestamp"].dt.strftime("%H:%M:%S")
        st.dataframe(history_df[["Time", "Price"]])

# Set up real-time updates using JavaScript
st.markdown(f"""
<script>
    function refreshData() {{
        window.parent.document.querySelector('[data-testid="stDecoration"]').dispatchEvent(new MouseEvent("click"));
        setTimeout(refreshData, {update_interval * 1000});
    }}
    setTimeout(refreshData, {update_interval * 1000});
</script>
""", unsafe_allow_html=True)