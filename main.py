import yfinance as yf
import pandas as pd
import mplfinance as mpf
import datetime
import os
from PIL import Image
import google.generativeai as genai
import matplotlib.pyplot as plt
import niftystocks.ns as ns
import csv

def configure_gemini():
    """Configure and return Gemini model with specific generation settings"""
    genai.configure(api_key="API KEY HERE")
    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-002",
        generation_config=generation_config,
    )
    
    return model

def upload_to_gemini(path, mime_type=None):
    """Upload file to Gemini and return file object"""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def get_gemini_prediction(model, image_path):
    """Get prediction and confidence from Gemini using uploaded image"""
    # Upload the image
    image_file = upload_to_gemini(image_path, mime_type="image/png")
    
    # Create chat session with initial prompt and image
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    image_file,
                ],
            }
        ]
    )
    
    # Updated prompt to request confidence level
    prompt = """
    Analyze the candlestick patterns in this graph and provide:
    1. A prediction (up, down, sideways, or don't know)
    2. Your confidence level (0.0 to 1.0)

    Not all graphs will have a up or downwards trend, so respond with 'sideways' unless you see a strong patterns.
    
    Respond in this format only:
    prediction|confidence
    Example: up|0.95 or down|0.85
    """
    
    # Get response
    response = chat_session.send_message(prompt)
    response_text = response.text.strip().lower()
    
    try:
        # Parse prediction and confidence
        prediction, confidence = response_text.split('|')
        confidence = float(confidence)
        
        # Return None if confidence is below threshold
        if confidence < 0.5:
            return prediction, confidence
        
        return prediction, confidence
    except:
        return None, 0

def get_nifty50_stocks():
    """Get Nifty 50 stocks"""
    # Using niftystocks library to get Nifty 50 constituents
    try:
        nifty50 = ns.get_nifty_next50_with_ns()
        return nifty50
    except:
        # Fallback to Nifty 100 if Nifty 200 is not available
        print("Warning: Unable to get Nifty 200 stocks, falling back to Nifty 100")
        return ns.get_nifty100_with_ns()

def plot_candlestick_chart(data, ticker, save_path='charts'):
    os.makedirs(save_path, exist_ok=True)
    
    # Create the candlestick chart
    fig, ax = mpf.plot(
        data,
        type='candle',
        volume=True,
        style='charles',
        title=f'{ticker} Candlestick Chart',
        returnfig=True
    )
    
    # Save the plot to a file
    chart_path = os.path.join(save_path, f'{ticker}_chart.png')
    fig.savefig(chart_path)
    plt.close(fig)
    return chart_path

def calculate_trade_parameters(current_price, trade_type):
    """Calculate stop loss and target prices based on 1:3 risk-reward ratio"""
    stop_loss_pct = 0.01  # 1% stop loss
    target_pct = 0.03     # 3% target
    
    if trade_type == "BUY":
        stop_loss = round(current_price * (1 - stop_loss_pct), 2)
        target = round(current_price * (1 + target_pct), 2)
    else:  # SELL
        stop_loss = round(current_price * (1 + stop_loss_pct), 2)
        target = round(current_price * (1 - target_pct), 2)
    
    return stop_loss, target

def print_trade_analysis(ticker, prediction, confidence, current_price):
    """Print trade analysis including confidence level"""
    if prediction and confidence >= 0.5:
        trade_type = "BUY" if prediction == "up" else "SELL"
        stop_loss, target = calculate_trade_parameters(current_price, trade_type)
        
        # Calculate percentages based on trade direction
        if trade_type == "BUY":
            sl_percentage = ((stop_loss - current_price) / current_price) * 100
            target_percentage = ((target - current_price) / current_price) * 100
        else:  # SELL
            sl_percentage = ((stop_loss - current_price) / current_price) * 100
            target_percentage = ((target - current_price) / current_price) * 100
        
        print(f"""
        =====================================
        Trade Analysis for {ticker}
        =====================================
        Signal: {trade_type}
        Confidence: {confidence:.2%}
        Current Price: {current_price:.2f}
        Stop Loss: {stop_loss:.2f} ({sl_percentage:.2f}%)
        Target: {target:.2f} ({target_percentage:.2f}%)
        Risk:Reward = 1:3
        =====================================
        """)
    else:
        print(f"""
        =====================================
        Trade Analysis for {ticker}
        =====================================
        No trade signal - {'Low confidence' if prediction else 'No clear prediction'}
        Current Price: {current_price:.2f}
        =====================================
        """)

def get_intraday_data(ticker, start_date, end_date, interval='15m'):
    """Get 15-minute intraday data for the specified period"""
    # Add buffer days to ensure we get enough 15-min candles
    buffer_start = start_date - datetime.timedelta(days=5)  # Add more buffer for 15-min data
    buffer_end = end_date + datetime.timedelta(days=1)
    
    try:
        data = yf.download(ticker, start=buffer_start, end=buffer_end, interval=interval)
        # Filter market hours (9:15 AM to 3:30 PM IST)
        data = data.between_time('09:15', '15:30')
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def analyze_intraday_trade_outcome(data, trade_type, entry_time, entry_price, stop_loss, target):
    """Analyze the outcome of an intraday trade"""
    realized_pnl = 0
    for timestamp, row in data.iterrows():
        if trade_type == "BUY":
            if row['Low'] <= stop_loss:
                realized_pnl = ((stop_loss - entry_price) / entry_price) * 100
                return f"Stop loss hit at {timestamp.strftime('%H:%M')}", realized_pnl
            if row['High'] >= target:
                realized_pnl = ((target - entry_price) / entry_price) * 100
                return f"Target hit at {timestamp.strftime('%H:%M')}", realized_pnl
        else:  # SELL
            if row['High'] >= stop_loss:
                realized_pnl = ((entry_price - stop_loss) / entry_price) * 100
                return f"Stop loss hit at {timestamp.strftime('%H:%M')}", realized_pnl
            if row['Low'] <= target:
                realized_pnl = ((entry_price - target) / entry_price) * 100
                return f"Target hit at {timestamp.strftime('%H:%M')}", realized_pnl
    
    # If neither stop loss nor target was hit, calculate P&L at last price
    last_price = data.iloc[-1]['Close']
    realized_pnl = ((last_price - entry_price) / entry_price * 100) if trade_type == "BUY" else \
                   ((entry_price - last_price) / entry_price * 100)
    return f"Squared off at market close, P&L: {realized_pnl:.2f}%", realized_pnl

def backtest_intraday_trades(start_date_str, end_date_str=None, stocks=None):
    """Backtest intraday trades with confidence threshold using 15-min candles"""
    try:
        model = configure_gemini()
        stocks = stocks or get_nifty50_stocks()
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else datetime.datetime.now()
        
        all_results = []
        all_analyses = []  # New list to store all analyses including low confidence ones
        current_date = start_date
        
        while current_date <= end_date:
            if current_date.weekday() >= 5:  # Skip weekends
                current_date += datetime.timedelta(days=1)
                continue
                
            print(f"\nAnalyzing trades for {current_date.date()}")
            
            for stock in stocks:
                try:
                    # Get 15-min data for analysis
                    analysis_start = current_date - datetime.timedelta(days=5)  # More buffer for 15-min data
                    analysis_end = current_date + datetime.timedelta(days=1)
                    data = get_intraday_data(stock, analysis_start, analysis_end)
                    
                    if data.empty:
                        print(f"No data available for {stock} on {current_date.date()}")
                        continue
                    
                    # Analyze every 15 minutes from market open
                    market_open = pd.Timestamp(current_date.date()) + pd.Timedelta(hours=9, minutes=15)
                    market_close = pd.Timestamp(current_date.date()) + pd.Timedelta(hours=15, minutes=30)
                    current_time = market_open
                    
                    while current_time <= market_close:
                        # Get data up to current time for analysis
                        analysis_data = data[:current_time].tail(25)
                        
                        if len(analysis_data) < 25:
                            current_time += pd.Timedelta(minutes=60)
                            continue
                        
                        # Get the entry price
                        try:
                            entry_price = data.loc[current_time, 'Open']
                        except KeyError:
                            current_time += pd.Timedelta(minutes=60)
                            continue
                        
                        # Generate and analyze chart
                        chart_path = plot_candlestick_chart(analysis_data, stock)
                        prediction, confidence = get_gemini_prediction(model, chart_path)
                        
                        # Calculate trade parameters regardless of confidence
                        trade_type = "BUY" if prediction == "up" else "SELL" if prediction == "down" else "NO TRADE"
                        stop_loss, target = calculate_trade_parameters(entry_price, trade_type) if trade_type != "NO TRADE" else (None, None)
                        
                        # Analyze outcome if it's a valid trade type
                        outcome = "No clear prediction"
                        realized_pnl = 0
                        if trade_type != "NO TRADE":
                            outcome, realized_pnl = analyze_intraday_trade_outcome(
                                data[current_time:],
                                trade_type,
                                current_time,
                                entry_price,
                                stop_loss,
                                target
                            )
                        
                        # Store analysis result
                        analysis_result = {
                            'date': current_date.date(),
                            'time': current_time.strftime('%H:%M'),
                            'stock': stock,
                            'prediction': prediction,
                            'confidence': confidence,
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'trade_type': trade_type,
                            'stop_loss': stop_loss,
                            'target': target,
                            'outcome': outcome,
                            'realized_pnl': realized_pnl
                        }
                        
                        all_analyses.append(analysis_result)
                        
                        # If high confidence trade, add to results
                        if prediction and confidence >= 0.5:
                            all_results.append(analysis_result)
                        
                        # Cleanup and move to next time interval
                        if 'chart_path' in locals():
                            os.remove(chart_path)
                        current_time += pd.Timedelta(minutes=60)
                
                except Exception as e:
                    print(f"Error processing {stock} on {current_date.date()}: {e}")
                    continue
            
            current_date += datetime.timedelta(days=1)
        
        # Print backtest summary and save results
        save_results_to_csv(all_analyses, start_date_str)  # Changed to save all_analyses
        print_backtest_summary(all_results, start_date_str)
        
    except Exception as e:
        print(f"Fatal error in backtest execution: {e}")

# Update the print_backtest_summary function to handle intraday results
def print_backtest_summary(results, start_date):
    """Print summary of backtest results with P&L analysis"""
    print(f"\n====================================")
    print(f"Intraday Backtest Results from {start_date}")
    print(f"====================================")
    
    total_trades = len(results)
    if total_trades == 0:
        print("No trades were generated during this period")
        return
    
    # Initialize counters and P&L tracking
    target_hits = 0
    sl_hits = 0
    squared_off = 0
    total_pnl = 0
    total_pnl_percentage = 0
    
    # Calculate statistics and P&L
    for result in results:
        outcome = result['outcome']
        entry_price = result['entry_price']
        trade_type = result['trade_type']
        
        if 'Target hit' in outcome:
            target_hits += 1
            pnl = result['target'] - entry_price if trade_type == "BUY" else entry_price - result['target']
            pnl_percentage = 3.0 if trade_type == "BUY" else 3.0
            
        elif 'Stop loss hit' in outcome:
            sl_hits += 1
            pnl = result['stop_loss'] - entry_price if trade_type == "BUY" else entry_price - result['stop_loss']
            pnl_percentage = -1.0 if trade_type == "BUY" else -1.0
            
        else:  # Squared off at day end
            squared_off += 1
            current_pnl = float(outcome.split("P&L: ")[1].rstrip("%"))
            pnl_percentage = current_pnl
            pnl = (entry_price * current_pnl / 100)
        
        total_pnl += pnl
        total_pnl_percentage += pnl_percentage
    
    # Print summary statistics
    print(f"\nTotal Trades: {total_trades}")
    print(f"Target Hits: {target_hits} ({(target_hits/total_trades)*100:.1f}%)")
    print(f"Stop Loss Hits: {sl_hits} ({(sl_hits/total_trades)*100:.1f}%)")
    print(f"Squared Off: {squared_off} ({(squared_off/total_trades)*100:.1f}%)")
    
    print(f"\nProfit/Loss Summary:")
    print(f"Total P&L: ₹{total_pnl:.2f}")
    print(f"Average P&L per trade: ₹{(total_pnl/total_trades):.2f}")
    print(f"Total P&L Percentage: {total_pnl_percentage:.2f}%")
    print(f"Average P&L Percentage per trade: {(total_pnl_percentage/total_trades):.2f}%")
    
    winning_trades = target_hits
    closed_trades = target_hits + sl_hits
    win_rate = (winning_trades / closed_trades) * 100 if closed_trades > 0 else 0
    print(f"\nWin Rate (excluding squared off trades): {win_rate:.1f}%")
    
    print("\nDetailed Trade Results:")
    print("----------------------------------------")
    for result in results:
        print(f"\nDate: {result['date']}")
        print(f"Stock: {result['stock']}")
        print(f"Trade Type: {result['trade_type']}")
        print(f"Entry Time: {result['entry_time'].strftime('%H:%M')}")
        print(f"Entry: {result['entry_price']:.2f}")
        print(f"Stop Loss: {result['stop_loss']:.2f}")
        print(f"Target: {result['target']:.2f}")
        print(f"Outcome: {result['outcome']}")
    print("----------------------------------------")

def save_results_to_csv(results, start_date):
    """Save backtest results to CSV file"""
    filename = f"nextnifty_backtest_results_{start_date}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Updated headers to include time and realized P&L
        writer.writerow(['Date', 'Time', 'Stock', 'Prediction', 'Confidence', 'Trade Type', 
                        'Entry Time', 'Entry Price', 'Stop Loss', 'Target', 
                        'Outcome', 'Realized P&L (%)'])
        # Write data
        for result in results:
            writer.writerow([
                result['date'],
                result['time'],
                result['stock'],
                result.get('prediction', 'N/A'),
                result.get('confidence', 0),
                result['trade_type'],
                result['entry_time'],
                result['entry_price'],
                result.get('stop_loss', 'N/A'),
                result.get('target', 'N/A'),
                result['outcome'],
                result['realized_pnl']
            ])
    print(f"\nResults saved to {filename}")

def live_trading(stocks=None):
    """Perform live trading analysis for today's date and print trades to be made"""
    try:
        model = configure_gemini()
        stocks = stocks or get_nifty50_stocks()
        today = datetime.datetime.now().date()
        
        print(f"\nLive Trading Analysis for {today}")
        
        for stock in stocks:
            try:
                # Get 15-min data for today's analysis
                data = get_intraday_data(stock, today, today)
                
                if data.empty:
                    print(f"No data available for {stock} today")
                    continue
                
                # Analyze every 15 minutes from market open
                market_open = pd.Timestamp(today) + pd.Timedelta(hours=9, minutes=15)
                market_close = pd.Timestamp(today) + pd.Timedelta(hours=15, minutes=30)
                current_time = market_open
                
                while current_time <= market_close:
                    # Get data up to current time for analysis
                    analysis_data = data[:current_time].tail(25)
                    
                    if len(analysis_data) < 25:
                        current_time += pd.Timedelta(minutes=60)
                        continue
                    
                    # Get the entry price
                    try:
                        entry_price = data.loc[current_time, 'Open']
                    except KeyError:
                        current_time += pd.Timedelta(minutes=60)
                        continue
                    
                    # Generate and analyze chart
                    chart_path = plot_candlestick_chart(analysis_data, stock)
                    prediction, confidence = get_gemini_prediction(model, chart_path)
                    
                    # Calculate trade parameters if confidence is high
                    if prediction and confidence >= 0.5:
                        trade_type = "BUY" if prediction == "up" else "SELL"
                        stop_loss, target = calculate_trade_parameters(entry_price, trade_type)
                        
                        # Print trade details
                        print(f"""
                        =====================================
                        Trade Signal for {stock} at {current_time.strftime('%H:%M')}
                        =====================================
                        Signal: {trade_type}
                        Confidence: {confidence:.2%}
                        Entry Price: {entry_price:.2f}
                        Stop Loss: {stop_loss:.2f}
                        Target: {target:.2f}
                        =====================================
                        """)
                    
                    # Cleanup and move to next time interval
                    if 'chart_path' in locals():
                        os.remove(chart_path)
                    current_time += pd.Timedelta(minutes=60)
            
            except Exception as e:
                print(f"Error processing {stock} today: {e}")
                continue
    
    except Exception as e:
        print(f"Fatal error in live trading execution: {e}")

if __name__ == "__main__":
    # For backtesting
    backtest_intraday_trades('2024-11-19', '2024-11-19')  # Test for specific period
    # For live trading
    # live_trading()  # Analyze trades for today