# AI-Powered Stock Trading System

## Overview
This project is an AI-powered stock trading system that uses Google's Gemini AI to analyze candlestick patterns and make trading predictions. The system can operate in two modes:
1. Real-time analysis of current market conditions
2. Backtesting mode to evaluate historical performance

## Features
- Analyzes candlestick patterns using Gemini AI
- Supports both long (BUY) and short (SELL) positions
- Implements 1:3 risk-reward ratio (1% stop loss, 3% target)
- Generates trade signals for Nifty 50's top 10 stocks
- Includes backtesting functionality with detailed performance metrics
- Automatic stop-loss and target calculation
- Comprehensive trade analysis and reporting

## Requirements

## Tools and Technologies

### Programming Languages
- Python 3.8+

### Key Libraries
- google-generativeai: For AI-powered pattern recognition
- yfinance: For fetching historical stock data
- pandas: For data manipulation and analysis
- mplfinance: For generating candlestick charts
- pillow: For image processing
- niftystocks: For accessing Nifty 50 stock information

### AI Model
- Google's Gemini 1.5 Flash: Advanced AI model for image analysis and pattern recognition

## Approach

### 1. Planning Phase
- Identified the need for AI-powered technical analysis
- Researched available AI models and chose Gemini for its advanced image recognition capabilities
- Designed the system architecture to support both real-time and backtesting modes
- Established risk management parameters (1:3 risk-reward ratio)

### 2. Development Phase

#### Data Collection Module
- Implemented yfinance integration for reliable market data
- Created functions to fetch and process historical price data
- Developed data validation and error handling mechanisms

#### Chart Generation System
- Built candlestick chart generation using mplfinance
- Optimized chart parameters for AI analysis
- Implemented automatic chart cleanup to manage storage

#### AI Integration
- Developed Gemini AI integration for pattern recognition
- Created a robust image upload and analysis pipeline
- Implemented error handling for API failures

#### Trade Logic
- Developed trade signal generation based on AI predictions
- Implemented automatic stop-loss and target calculation
- Created position sizing logic based on risk parameters

#### Backtesting Engine
- Built comprehensive backtesting functionality
- Implemented trade outcome analysis
- Developed performance metrics calculation
- Created detailed reporting system

### 3. Testing Phase

#### Unit Testing
- Tested individual components:
  - Data fetching reliability
  - Chart generation accuracy
  - AI prediction consistency
  - Trade calculation precision

#### Integration Testing
- Verified system components work together:
  - Data flow from fetch to analysis
  - AI integration with trade logic
  - Backtesting accuracy

#### Performance Testing
- Conducted backtests on historical data
- Analyzed system response times
- Optimized resource usage

### Algorithms and Frameworks

#### Pattern Recognition
- Uses Gemini AI's computer vision capabilities
- Analyzes candlestick patterns and market structure
- Generates directional predictions (up, down, sideways)

#### Trade Management
- Entry Calculation:
  - Based on current market price
  - Considers market volatility
  - Implements position sizing rules

- Risk Management:
  - Fixed 1% stop-loss calculation
  - 3% target setting
  - 1:3 risk-reward ratio enforcement

#### Performance Analysis
- Win Rate Calculation:
  ```python
  win_rate = (target_hits / (target_hits + stop_loss_hits)) * 100
  ```

- P&L Calculation:
  ```python
  trade_pnl = (exit_price - entry_price) / entry_price * 100
  ```

- Risk-Adjusted Returns:
  - Considers both winning and losing trades
  - Accounts for open position P&L
  - Calculates average return per trade

### System Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│  Data Fetching  │ ──► │  Chart Gen   │ ──► │  AI Analysis   │
└─────────────────┘     └──────────────┘     └────────────────┘
                                                     │
┌─────────────────┐     ┌──────────────┐           ▼
│    Reporting    │ ◄── │ Trade Logic  │ ◄── ┌────────────────┐
└─────────────────┘     └──────────────┘     │  Prediction    │
                                             └────────────────┘
```

### Future Development
- Implement machine learning for pattern recognition enhancement
- Add more sophisticated position sizing algorithms
- Develop real-time alert system
- Create web interface for easier interaction
- Add portfolio management capabilities
- Implement more advanced backtesting scenarios

## Project Design & Development

### System Architecture Details

#### 1. Data Collection Layer
- **YFinance Integration**
  - Fetches real-time and historical stock data
  - Handles data validation and cleaning
  - Manages API rate limits and connection errors
  ```python
  data = yf.download(stock, start=start_date, end=end_date, interval='1d')
  ```

- **Data Processing**
  - Converts raw data to candlestick format
  - Implements data normalization
  - Handles missing data points
  ```python
  data = data.tail(25)  # Last 25 candles for analysis
  ```

#### 2. Chart Generation Layer
- **MPLFinance Integration**
  - Creates professional-grade candlestick charts
  - Customizes chart appearance for AI analysis
  - Manages temporary file storage
  ```python
  fig, ax = mpf.plot(
      data,
      type='candle',
      volume=True,
      style='charles',
      title=f'{ticker} Candlestick Chart',
      returnfig=True
  )
  ```

#### 3. AI Analysis Layer
- **Gemini AI Integration**
  - Handles image upload and processing
  - Manages API communication
  - Processes AI responses
  ```python
  chat_session = model.start_chat(
      history=[{
          "role": "user",
          "parts": [image_file],
      }]
  )
  ```

#### 4. Trade Logic Layer
- **Signal Generation**
  - Interprets AI predictions
  - Generates trade signals
  - Implements risk management rules

- **Position Management**
  - Calculates entry points
  - Sets stop-loss levels
  - Determines profit targets
  ```python
  def calculate_trade_parameters(current_price, trade_type):
      stop_loss_pct = 0.01  # 1% stop loss
      target_pct = 0.03     # 3% target
      # Calculate based on trade direction
  ```

#### 5. Analysis & Reporting Layer
- **Performance Metrics**
  - Calculates win rates
  - Tracks P&L
  - Generates performance reports

- **Backtesting Engine**
  - Simulates historical trades
  - Analyzes outcomes
  - Provides detailed statistics

### Component Details

#### 1. Core Components

##### Data Manager
- **Purpose**: Handles all data-related operations
- **Key Features**:
  - Real-time data fetching
  - Historical data processing
  - Data validation
- **Implementation**:
  ```python
  class DataManager:
      def get_historical_data(self, ticker, start_date, end_date):
          return yf.download(ticker, start=start_date, end=end_date)
  ```

##### Chart Generator
- **Purpose**: Creates visual representations of stock data
- **Key Features**:
  - Candlestick chart creation
  - Technical indicator overlay
  - Chart optimization for AI
- **Implementation**:
  ```python
  def plot_candlestick_chart(data, ticker, save_path='charts'):
      # Chart generation logic
  ```

##### AI Analyzer
- **Purpose**: Processes charts and generates predictions
- **Key Features**:
  - Image processing
  - Pattern recognition
  - Signal generation
- **Implementation**:
  ```python
  def get_gemini_prediction(model, image_path):
      # AI analysis logic
  ```

##### Trade Manager
- **Purpose**: Handles trade-related calculations and decisions
- **Key Features**:
  - Entry/exit calculations
  - Risk management
  - Position sizing
- **Implementation**:
  ```python
  def calculate_trade_parameters(current_price, trade_type):
      # Trade parameter calculations
  ```

#### 2. Supporting Components

##### Configuration Manager
- Handles API keys and system settings
- Manages environment variables
- Controls system parameters

##### Error Handler
- Manages exceptions
- Implements retry mechanisms
- Logs system errors

##### Performance Analyzer
- Calculates trading metrics
- Generates performance reports
- Tracks system efficiency

### Implementation Details

#### 1. Development Workflow
1. **Data Pipeline**
   - Implemented robust data fetching
   - Added data validation
   - Created error handling

2. **Chart Generation**
   - Optimized chart parameters
   - Implemented cleanup mechanisms
   - Added error handling

3. **AI Integration**
   - Developed API communication
   - Implemented response parsing
   - Added retry mechanisms

4. **Trade Logic**
   - Implemented risk management
   - Added position sizing
   - Created trade validation

#### 2. Key Technical Decisions

- **Choice of Libraries**
  - yfinance: Reliable, free data source
  - mplfinance: Professional charts
  - Gemini AI: Advanced pattern recognition

- **Architecture Decisions**
  - Modular design for maintainability
  - Separation of concerns
  - Error handling at each layer

- **Performance Optimizations**
  - Efficient data processing
  - Memory management
  - API rate limiting

#### 3. Code Organization
```
project/
├── gemini_chart.py     # Main trading logic
├── README.md          # Documentation
└── requirements.txt   # Dependencies
```

### Data Flow
1. User initiates analysis
2. System fetches stock data
3. Generates candlestick chart
4. Submits to Gemini AI
5. Processes prediction
6. Calculates trade parameters
7. Generates trade signals
8. Reports results

[Rest of the README remains the same...]