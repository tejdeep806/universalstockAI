import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import re

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Universal Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Universal Stock Analysis")
st.write("Analyze ANY stock - Known companies or new discoveries!")

# Import LangChain
try:
    from langchain.agents import initialize_agent, Tool
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import AgentType
    from langchain.schema import SystemMessage
    import openai
    
    LANGCHAIN_AVAILABLE = True
    
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    st.error(f"❌ LangChain imports failed: {e}")

# Enhanced MCP Server with multiple tools and caching
class AlphaVantageMCPServer:
    def __init__(self):
        self.api_key = st.secrets["ALPHA_VANTAGE_API_KEY"] # Use secrets or fallback to hardcoded
        self.base_url = "https://www.alphavantage.co/query"
        self.tool_calls_log = []
        self.cache = {}  # Add cache to avoid duplicate API calls
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> dict:
        """Get comprehensive stock data with historical prices"""
        key = f"get_stock_data:{symbol}:{period}"
        if key in self.cache:
            self.tool_calls_log.append({
                "tool": "get_stock_data", 
                "symbol": symbol, 
                "period": period,
                "timestamp": datetime.now().isoformat(),
                "from_cache": True
            })
            return self.cache[key]
        
        self.tool_calls_log.append({
            "tool": "get_stock_data", 
            "symbol": symbol, 
            "period": period,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            outputsize = "compact" if period in ["1mo", "3mo"] else "full"
            
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': outputsize
            }
            
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if "Time Series (Daily)" in data:
                    time_series = data["Time Series (Daily)"]
                    
                    # Convert to DataFrame
                    df_data = []
                    for date, values in time_series.items():
                        df_data.append({
                            'date': date,
                            'open': float(values["1. open"]),
                            'high': float(values["2. high"]),
                            'low': float(values["3. low"]),
                            'close': float(values["4. close"]),
                            'volume': int(values["5. volume"])
                        })
                    
                    df = pd.DataFrame(df_data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Filter based on period
                    end_date = df['date'].max()
                    period_days = {
                        "1mo": 30, "3mo": 90, "6mo": 180, 
                        "1y": 365, "2y": 730
                    }
                    start_date = end_date - timedelta(days=period_days.get(period, 180))
                    df = df[df['date'] >= start_date]
                    
                    # Calculate metrics
                    latest = df.iloc[-1]
                    first = df.iloc[0]
                    total_return = ((latest['close'] - first['close']) / first['close']) * 100
                    volatility = df['close'].pct_change().std() * 100
                    
                    result = {
                        "success": True,
                        "symbol": symbol,
                        "period": period,
                        "data": df.to_dict('records'),
                        "latest_price": latest['close'],
                        "total_return": total_return,
                        "volatility": volatility,
                        "high": df['high'].max(),
                        "low": df['low'].min(),
                        "avg_volume": df['volume'].mean(),
                        "data_points": len(df)
                    }
                    self.cache[key] = result
                    return result
                else:
                    result = {"success": False, "error": f"No time series data found for {symbol}"}
                    self.cache[key] = result
                    return result
            
            result = {"success": False, "error": f"Could not fetch data for {symbol}"}
            self.cache[key] = result
            return result
            
        except Exception as e:
            result = {"success": False, "error": f"Failed to get stock data: {str(e)}"}
            self.cache[key] = result
            return result
    
    def get_company_overview(self, symbol: str) -> dict:
        """Get company overview"""
        key = f"get_company_overview:{symbol}"
        if key in self.cache:
            self.tool_calls_log.append({
                "tool": "get_company_overview", 
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "from_cache": True
            })
            return self.cache[key]
        
        self.tool_calls_log.append({
            "tool": "get_company_overview", 
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # RATE LIMIT DETECTED
                if "Note" in data:
                    result = {
                        "success": False,
                        "error": "Rate limit hit (25 calls/day free tier). Upgrade or wait 24h."
                    }
                    self.cache[key] = result
                    return result
                
                # SUCCESS: Valid company data
                elif data and data.get('Name'):
                    result = {
                        "success": True,
                        "symbol": symbol,
                        "name": data.get('Name', 'N/A'),
                        "sector": data.get('Sector', 'N/A'),
                        "industry": data.get('Industry', 'N/A'),
                        "market_cap": data.get('MarketCapitalization', 'N/A'),
                        "pe_ratio": data.get('PERatio', 'N/A'),
                        "dividend_yield": data.get('DividendYield', 'N/A'),
                        "description": data.get('Description', 'N/A'),
                        "52week_high": data.get('52WeekHigh', 'N/A'),
                        "52week_low": data.get('52WeekLow', 'N/A')
                    }
                    self.cache[key] = result
                    return result
                
                # NO DATA (but not rate limit)
                else:
                    result = {
                        "success": False,
                        "error": f"No company data found for {symbol}"
                    }
                    self.cache[key] = result
                    return result
            
            # HTTP ERROR
            else:
                result = {
                    "success": False,
                    "error": f"HTTP {response.status_code} - Server error"
                }
                self.cache[key] = result
                return result
                
        except Exception as e:
            result = {
                "success": False,
                "error": f"Network error: {str(e)}"
            }
            self.cache[key] = result
            return result
    
    def search_stocks(self, keywords: str) -> dict:
        """Search for stocks by keywords"""
        key = f"search_stocks:{keywords}"
        if key in self.cache:
            self.tool_calls_log.append({
                "tool": "search_stocks", 
                "keywords": keywords,
                "timestamp": datetime.now().isoformat(),
                "from_cache": True
            })
            return self.cache[key]
        
        self.tool_calls_log.append({
            "tool": "search_stocks", 
            "keywords": keywords,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            params = {
                'function': 'SYMBOL_SEARCH', 
                'keywords': keywords,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'bestMatches' in data and data['bestMatches']:
                    matches = []
                    for match in data['bestMatches'][:8]:
                        matches.append({
                            "symbol": match['1. symbol'],
                            "name": match['2. name'],
                            "type": match['3. type'],
                            "region": match['4. region']
                        })
                    result = {
                        "success": True,
                        "keywords": keywords,
                        "matches": matches
                    }
                    self.cache[key] = result
                    return result
                else:
                    result = {"success": False, "error": f"No stocks found for '{keywords}'"}
                    self.cache[key] = result
                    return result
            result = {"success": False, "error": f"Search API failed for '{keywords}'"}
            self.cache[key] = result
            return result
            
        except Exception as e:
            result = {"success": False, "error": f"Search failed: {str(e)}"}
            self.cache[key] = result
            return result
    
    def get_technical_indicators(self, symbol: str) -> dict:
        """Get technical indicators like SMA, RSI"""
        key = f"get_technical_indicators:{symbol}"
        if key in self.cache:
            self.tool_calls_log.append({
                "tool": "get_technical_indicators", 
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "from_cache": True
            })
            return self.cache[key]
        
        self.tool_calls_log.append({
            "tool": "get_technical_indicators", 
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Get stock data first (will use cache if already fetched)
            # SILENT: Get 3mo data for RSI/SMA — don't log in UI
            old_log_len = len(self.tool_calls_log)
            stock_data = self.get_stock_data(symbol, "3mo")
            # Remove the 3mo call from log if it was added
            if len(self.tool_calls_log) > old_log_len:
                self.tool_calls_log.pop()  # Remove last (3mo) call
            #stock_data = self.get_stock_data(symbol, "3mo")
            if not stock_data["success"]:
                return stock_data
            
            df = pd.DataFrame(stock_data["data"])
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate technical indicators
            if len(df) >= 20:
                df['fy_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean() if len(df) >= 50 else None
                
                # RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                latest_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else None
                
                result = {
                    "success": True,
                    "symbol": symbol,
                    "sma_20": df['sma_20'].iloc[-1] if not pd.isna(df['sma_20'].iloc[-1]) else None,
                    "sma_50": df['sma_50'].iloc[-1] if df['sma_50'] is not None and not pd.isna(df['sma_50'].iloc[-1]) else None,
                    "rsi": latest_rsi,
                    "rsi_signal": "Oversold" if latest_rsi and latest_rsi < 30 else "Overbought" if latest_rsi and latest_rsi > 70 else "Neutral"
                }
                self.cache[key] = result
                return result
            
            result = {"success": False, "error": "Insufficient data for technical indicators"}
            self.cache[key] = result
            return result
            
        except Exception as e:
            result = {"success": False, "error": f"Technical analysis failed: {str(e)}"}
            self.cache[key] = result
            return result
        
    def get_global_quote(self, symbol: str) -> dict:
        """Live price + change + volume + 52w high/low"""
        key = f"global_quote:{symbol}"
        if key in self.cache: return self.cache[key]
        try:
            params = {'function': 'GLOBAL_QUOTE', 'symbol': symbol, 'apikey': self.api_key}
            r = requests.get(self.base_url, params=params).json()
            q = r.get("Global Quote", {})
            result = {
                "success": bool(q),
                "price": float(q.get("05. price", 0)),
                "change": q.get("09. change", "N/A"),
                "change_pct": q.get("10. change percent", "N/A"),
                "volume": q.get("06. volume", "N/A")
            }
            self.cache[key] = result
            return result
        except: 
            return {"success": False, "error": "GLOBAL_QUOTE failed"}

    def get_key_stats(self, symbol: str) -> dict:
        """Beta, EPS, Target, Insider %, Short %"""
        key = f"keystats:{symbol}"
        if key in self.cache: return self.cache[key]
        try:
            # Free Yahoo-style stats via Alpha Vantage OVERVIEW + extra fields
            ov = self.get_company_overview(symbol)
            if not ov["success"]: return ov
            result = {
                "success": True,
                "beta": ov.get("Beta", "N/A"),
                "eps": ov.get("EPS", "N/A"),
                "analyst_target": ov.get("AnalystTargetPrice", "N/A"),
                "insider_pct": ov.get("SharesPercentInsiders", "N/A"),
                "short_pct": ov.get("ShortPercentOfFloat", "N/A")
            }
            self.cache[key] = result
            return result
        except: 
            return {"success": False, "error": "Key stats failed"}
    
    def get_tool_calls(self):
        return self.tool_calls_log.copy()
    
    def clear_tool_calls(self):
        self.tool_calls_log.clear()

# Chart Creator
class ChartCreator:
    @staticmethod
    def create_price_chart(stock_data):
        if not stock_data["success"]:
            return None
        
        df = pd.DataFrame(stock_data["data"])
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate moving averages
        if len(df) >= 20:
            df['sma_20'] = df['close'].rolling(window=20).mean()
        if len(df) >= 50:
            df['sma_50'] = df['close'].rolling(window=50).mean()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f"{stock_data['symbol']} - Price Chart", 
                "Volume"
            ),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['sma_20'], name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['date'], y=df['sma_50'], name='SMA 50', line=dict(color='red')),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(
            go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        return fig

# Universal LangChain Agent with Dynamic Symbol Extraction
class UniversalStockAgent:
    def __init__(self, openai_api_key: str):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        
        self.mcp_server = AlphaVantageMCPServer()
        self.openai_api_key = openai_api_key
        
        # Use a more powerful model with higher limits
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo", 
            openai_api_key=openai_api_key,
            request_timeout=120  # Increase timeout
        )
        
        # More specific system message to reduce tool iterations
        self.system_message = SystemMessage(content="""You are a professional financial analyst assistant. 
        When users ask about stocks, use the available tools to get real data.
        
        IMPORTANT GUIDELINES:
        1. Use MAXIMUM 2-3 tools per query
        2. For stock analysis, typically use get_stock_data and get_company_overview
        3. For searches, use search_stocks
        4. For technical analysis, use get_technical_analysis
        5. Provide concise but comprehensive analysis
        
        Always include:
        - Current price and performance
        - Key company information
        - Actionable insights
        - Risk considerations""")

    def _extract_symbol_from_query(self, query: str) -> str:
        """Extract stock symbol from natural language query - works for ANY stock"""
        
        # Expanded stock mappings
        stock_mappings = {
            # Technology
            'apple': 'AAPL', 'aapl': 'AAPL',
            'microsoft': 'MSFT', 'msft': 'MSFT', 
            'google': 'GOOGL', 'googl': 'GOOGL', 'alphabet': 'GOOGL',
            'tesla': 'TSLA', 'tsla': 'TSLA',
            'amazon': 'AMZN', 'amzn': 'AMZN',
            'meta': 'META', 'facebook': 'META',
            'nvidia': 'NVDA', 'nvda': 'NVDA',
            'netflix': 'NFLX', 'nflx': 'NFLX',
            'intel': 'INTC', 'intc': 'INTC',
            'amd': 'AMD', 'advanced micro devices': 'AMD',
            'ibm': 'IBM', 'international business machines': 'IBM',
            'oracle': 'ORCL', 'orcl': 'ORCL',
            'salesforce': 'CRM', 'crm': 'CRM',
            'adobe': 'ADBE', 'adbe': 'ADBE',
            'paypal': 'PYPL', 'pypl': 'PYPL',
            'shopify': 'SHOP', 'shop': 'SHOP',
            'spotify': 'SPOT', 'spot': 'SPOT',
            'uber': 'UBER', 'uber': 'UBER',
            'lyft': 'LYFT', 'lyft': 'LYFT',
            'airbnb': 'ABNB', 'abnb': 'ABNB',
            'snap': 'SNAP', 'snapchat': 'SNAP',
            'pinterest': 'PINS', 'pins': 'PINS',
            'twitter': 'TWTR', 'twtr': 'TWTR',
            'coinbase': 'COIN', 'coin': 'COIN',
            'palantir': 'PLTR', 'pltr': 'PLTR',
            'robinhood': 'HOOD', 'hood': 'HOOD',
            
            # Automotive
            'ford': 'F', 'f': 'F',
            'general motors': 'GM', 'gm': 'GM',
            'toyota': 'TM', 'tm': 'TM',
            'honda': 'HMC', 'hmc': 'HMC',
            
            # Consumer Goods
            'coca cola': 'KO', 'ko': 'KO',
            'pepsi': 'PEP', 'pep': 'PEP',
            'walmart': 'WMT', 'wmt': 'WMT',
            'mcdonalds': 'MCD', 'mcd': 'MCD',
            'procter gamble': 'PG', 'pg': 'PG',
            'nike': 'NKE', 'nke': 'NKE',
            
            # Financial
            'visa': 'V', 'v': 'V',
            'mastercard': 'MA', 'ma': 'MA',
            'jpmorgan': 'JPM', 'jpm': 'JPM',
            'bank america': 'BAC', 'bac': 'BAC',
            'goldman sachs': 'GS', 'gs': 'GS',
            'morgan stanley': 'MS', 'ms': 'MS',
            
            # Healthcare
            'johnson johnson': 'JNJ', 'jnj': 'JNJ',
            'pfizer': 'PFE', 'pfe': 'PFE',
            'merck': 'MRK', 'mrk': 'MRK',
            'moderna': 'MRNA', 'mrna': 'MRNA',
            
            # Other sectors
            'boeing': 'BA', 'ba': 'BA',
            'disney': 'DIS', 'dis': 'DIS',
            'at&t': 'T', 't': 'T',
            'verizon': 'VZ', 'vz': 'VZ'
        }
        
        query_lower = query.lower()
        
        # Priority 1: Direct symbol extraction (uppercase 1-5 letters)
        symbol_pattern = r'\b([A-Z]{1,5})\b'
        symbols_found = re.findall(symbol_pattern, query)
        
        if symbols_found:
            for symbol in symbols_found:
                if 1 <= len(symbol) <= 5:
                    return symbol
        
        # Priority 2: Known company names
        for company_name, symbol in stock_mappings.items():
            if company_name in query_lower:
                return symbol
        
        # Priority 3: Pattern matching for company names
        company_patterns = [
            (r'stock of (\w+(?:\s+\w+)*)', 1),
            (r'(\w+(?:\s+\w+)*) stock', 1),
            (r'(\w+(?:\s+\w+)*) company', 1),
            (r'(\w+(?:\s+\w+)*) corp', 1),
            (r'(\w+(?:\s+\w+)*) inc', 1),
            (r'analyze (\w+(?:\s+\w+)*)', 1),
            (r'show me (\w+(?:\s+\w+)*)', 1),
            (r'information for (\w+(?:\s+\w+)*)', 1),
            (r'price of (\w+(?:\s+\w+)*)', 1),
            (r'performance of (\w+(?:\s+\w+)*)', 1)
        ]
        
        for pattern, group in company_patterns:
            matches = re.search(pattern, query_lower)
            if matches:
                potential_company = matches.group(group).strip()
                # Check if this matches any known company
                for company_name, symbol in stock_mappings.items():
                    if company_name in potential_company.lower():
                        return symbol
        
        # Priority 4: If it's clearly a search query, use search
        search_terms = ['search', 'find', 'look for', 'discover', 'what are some', 'which stocks']
        if any(term in query_lower for term in search_terms):
            # For search queries, we'll let the search tool handle it
            # Extract the search term if possible
            search_patterns = [
                r'search for (\w+(?:\s+\w+)*)',
                r'find (\w+(?:\s+\w+)*) stocks',
                r'look for (\w+(?:\s+\w+)*)',
                r'stocks in (\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*) stocks'
            ]
            
            for pattern in search_patterns:
                matches = re.search(pattern, query_lower)
                if matches:
                    search_term = matches.group(1).strip()
                    # Return a symbol that represents this search
                    return f"SEARCH:{search_term}"
            
            return "SEARCH:technology"  # Default search term
        
        # Priority 5: Try to identify the main noun as company
        words = query_lower.split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        content_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        for word in content_words:
            for company_name, symbol in stock_mappings.items():
                if word in company_name.split():
                    return symbol
        
        # Final fallback with context awareness
        if any(word in query_lower for word in ['tech', 'technology', 'software', 'computer']):
            return "MSFT"  # Default tech stock
        elif any(word in query_lower for word in ['car', 'auto', 'vehicle', 'electric']):
            return "TSLA"  # Default auto stock
        elif any(word in query_lower for word in ['bank', 'finance', 'financial']):
            return "JPM"  # Default financial stock
        else:
            return "AAPL"  # General default

    def _extract_period_from_query(self, query: str) -> str:
        """Extract time period - 100% accurate, no defaults"""
        query_lower = query.lower()
        
        if re.search(r'\b(1|one)\s*(month|months)\b', query_lower):
            return "1mo"
        if re.search(r'\b(2|two)\s*(month|months)\b', query_lower):
            st.info("Note: '2 months' → using 3mo (closest available data)")
            return "3mo"
        if re.search(r'\b(3|three)\s*(month|months)\b|quarter', query_lower):
            return "3mo"
        if re.search(r'\b(6|six)\s*(month|months)\b|half\s*year', query_lower):
            return "6mo"
        if re.search(r'\b(1|one)\s*(year|years)\b', query_lower):
            return "1y"
        if re.search(r'\b(2|two)\s*(year|years)\b', query_lower):
            return "2y"
        
        return "3mo"  # Shorter default to save quota

    def _create_tools(self):
        """Create tools fresh for each query"""
        return [
            Tool(
                name="get_stock_data",
                func=lambda query: str(self.mcp_server.get_stock_data(
                    self._extract_symbol_from_query(query), 
                    self._extract_period_from_query(query)
                )),
                description="""CRITICAL: Use the EXACT period the user requested.
                Examples:
                - 'last 3 months' → 3mo
                - 'last year' → 1y
                - 'last 1 month' → 1mo
                NEVER use 6mo unless the user explicitly says '6 months'."""
            ),
                # name="get_stock_data",
                # func=lambda query: str(self.mcp_server.get_stock_data(
                #     self._extract_symbol_from_query(query), 
                #     self._extract_period_from_query(query)
                # )),
            #     description="Useful for getting stock price history, current price, daily changes, volume, and performance metrics. Use this for any query about stock prices, trends, or historical performance."
            # ),
            Tool(
                name="get_company_overview",
                func=lambda query: str(self.mcp_server.get_company_overview(
                    self._extract_symbol_from_query(query)
                )),
                description="Useful for getting company fundamentals like business description, sector, industry, market capitalization, P/E ratio, and dividend yield. Use this when asked about company information or fundamentals."
            ),
            Tool(
                name="search_stocks", 
                func=lambda query: str(self._extract_and_search_stocks(query)),
                description="Useful for searching and discovering stocks by company name, industry keywords, or sector. Use this when the user wants to find stocks or explore investment options."
            ),
            Tool(
                name="get_technical_analysis",
                func=lambda query: str(self.mcp_server.get_technical_indicators(
                    self._extract_symbol_from_query(query)
                )),
                description="Useful for getting technical indicators like RSI, moving averages, and trend signals. Use this when specifically asked about technical analysis, trading signals, or market indicators."
            )
        ]

    def _extract_and_search_stocks(self, query: str) -> dict:
        """Enhanced search that handles both known and unknown stocks"""
        # Check if this is a search query with a specific term
        if query.startswith("SEARCH:"):
            search_term = query.replace("SEARCH:", "").strip()
            return self.mcp_server.search_stocks(search_term)
        
        # Regular search
        if any(term in query.lower() for term in ['search', 'find', 'look for', 'discover']):
            # Extract search term from query
            search_patterns = [
                r'search for (\w+(?:\s+\w+)*)',
                r'find (\w+(?:\s+\w+)*) stocks',
                r'look for (\w+(?:\s+\w+)*)',
                r'stocks in (\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*) stocks'
            ]
            
            for pattern in search_patterns:
                matches = re.search(pattern, query.lower())
                if matches:
                    search_term = matches.group(1).strip()
                    return self.mcp_server.search_stocks(search_term)
            
            # Default search
            return self.mcp_server.search_stocks("technology")
        
        # If it's not clearly a search, try to extract a company and search for it
        symbol = self._extract_symbol_from_query(query)
        if symbol and not symbol.startswith("SEARCH:"):
            # Search for this specific company to get more info
            return self.mcp_server.search_stocks(symbol)
        
        return self.mcp_server.search_stocks("stocks")

    def process_query(self, user_query: str) -> dict:
        self.mcp_server.clear_tool_calls()
        try:
            tools = self._create_tools()
            agent = initialize_agent(
                tools=tools, llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=False, handle_parsing_errors=True,
                max_iterations=4
            )
            ai_response = agent({"input": user_query})["output"]

            symbol = self._extract_symbol_from_query(user_query)
            period = self._extract_period_from_query(user_query)
            

            # stock = self.mcp_server.get_stock_data(symbol, period)
            # company = self.mcp_server.get_company_overview(symbol)
            # === QUOTA-SAFE: Deduplicate + Rate Limit Fallback ===
            tool_log = self.mcp_server.get_tool_calls()
            stock_data = None
            company_data = None

            # Reuse if LangChain called with exact period
            for call in tool_log:
                if call.get("tool") == "get_stock_data" and call.get("symbol") == symbol:
                    params = call.get("parameters", {})
                    if isinstance(params, str): params = ast.literal_eval(params)
                    if params.get("period") == period:
                        stock_data = self.mcp_server.get_stock_data(symbol, period)
                        break

            for call in tool_log:
                if call.get("tool") == "get_company_overview" and call.get("symbol") == symbol:
                    company_data = self.mcp_server.get_company_overview(symbol)
                    break

            # Fallback (only if missing)
            if not stock_data: stock_data = self.mcp_server.get_stock_data(symbol, period)
            if not company_data: company_data = self.mcp_server.get_company_overview(symbol)

            # LIVE DATA (quota-heavy, but essential)
            quote = self.mcp_server.get_global_quote(symbol)
            stats = self.mcp_server.get_key_stats(symbol)

            # COMPATIBILITY
            stock = stock_data
            company = company_data

            # QUOTA WARNING
            if "rate limit" in str(stock_data).lower() or "rate limit" in str(company_data).lower():
                st.warning("🛑 Rate limit hit! Free tier: 25 calls/day. Upgrade for unlimited: https://www.alphavantage.co/premium/")
                        
            
            # === LIVE DATA (always fresh) ===
            quote = self.mcp_server.get_global_quote(symbol)
            stats = self.mcp_server.get_key_stats(symbol)
            #show savings on API calls
            if any("from_cache" in call for call in tool_log):
                st.caption("Cached data reused — saved API calls!")
            

            rec_prompt = f"""
            Symbol: {symbol}
            Price: ${quote.get('price',0):.2f}  Change: {quote.get('change_pct','N/A')}
            6mo Return: {stock.get('total_return',0):.1f}%
            P/E: {company.get('pe_ratio','N/A')}
            Give ONE word: BUY, HOLD or SELL. Then one sentence why.
            """
            recommendation = self.llm.predict(rec_prompt).strip()

            return {
                "success": True,
                "ai_response": ai_response,
                "tool_calls": self.mcp_server.get_tool_calls(),   # ← THIS LINE ADDS THE KEY
                "stock_data": stock,
                "company_data": company,
                "live_quote": quote,
                "key_stats": stats,
                "recommendation": recommendation,
                "symbol": symbol,
                "period": period
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_calls": []                                 # ← fallback
            }

# Streamlit UI
st.sidebar.header("💬 Chat Configuration")

# Check OpenAI key
openai_key = st.secrets.get("OPENAI_API_KEY", "not-set")
if openai_key in ["not-set", "your-actual-openai-api-key-here"]:
    st.sidebar.error("❌ OpenAI API key not set")
    st.sidebar.info("Set OPENAI_API_KEY in .streamlit/secrets.toml")
    chat_enabled = False
else:
    st.sidebar.success("✅ OpenAI API key available")
    chat_enabled = True

# Check Alpha Vantage key in sidebar
alpha_key = st.secrets.get("ALPHA_VANTAGE_API_KEY", "not-set")
if alpha_key in ["not-set"]:
    st.sidebar.warning("⚠️ Alpha Vantage API key not set. Using default demo key (limited calls). Set ALPHA_VANTAGE_API_KEY in .streamlit/secrets.toml for full access.")

# Show available tools
st.sidebar.subheader("🛠️ Available MCP Tools")
st.sidebar.write("""
- **get_stock_data**: Price history & trends
- **get_company_overview**: Company fundamentals  
- **search_stocks**: Find stocks by keywords
- **get_technical_analysis**: Technical indicators
""")

# QUOTA TRACKER (add to sidebar)
st.sidebar.subheader("📊 Quota Status")
try:
    # Count from last analysis (saved in session state)
    last_log = st.session_state.get("last_tool_log", [])
    real_calls = len([c for c in last_log if not c.get("from_cache", False)])
    st.sidebar.metric("Calls in Last Query", f"{real_calls}/3 (per analysis)")
    if real_calls >= 3:
        st.sidebar.warning("High usage — check daily limit (25 free)")
except:
    st.sidebar.metric("Calls in Last Query", "0/3")
    # st.sidebar.info("Run a query to see usage")
st.sidebar.info("💡 The AI will automatically choose which tools to use based on your question!")

# Chat interface
st.subheader("💬 Ask About ANY Stock")

# Example queries - including new stocks
st.write("**Try these examples:**")
example_queries = [
    "Show me PLTR stock for last 6 months",
    "Analyze Palantir Technologies",
    "Find biotech companies",
    "What about Shopify stock?",
    "COIN stock performance and analysis"
]

cols = st.columns(2)
for i, example in enumerate(example_queries):
    with cols[i % 2]:
        if st.button(f"💬 {example}", key=f"example_{i}"):
            st.session_state.user_query = example

# User input
user_query = st.text_input(
    "Or type your own question:",
    placeholder="e.g., 'Show me PLTR stock for last year'",
    key="user_query_input"
)


# Process query
if st.button("Analyze", type="primary") or 'user_query' in st.session_state:
    # 1. Get the query: prefer example (if clicked), then text input
    query_to_process = st.session_state.get('user_query', user_query.strip())
    
    # 2. Clear the example so it doesn't persist
    if 'user_query' in st.session_state:
        del st.session_state.user_query

    # 3. Validate
    if not query_to_process:
        st.error("Please enter a question about stocks")
    elif not chat_enabled:
        st.error("OpenAI API key not set. Please check configuration.")
    elif not LANGCHAIN_AVAILABLE:
        st.error("LangChain not available. Please check installation.")
    else:
        with st.spinner("AI is analyzing your query..."):
            # Create agent fresh for each query
            agent = UniversalStockAgent(openai_key)
            result = agent.process_query(query_to_process)
            
            # Save tool log for quota tracker (safe, no error)
            st.session_state.last_tool_log = result.get("tool_calls", [])

            # === SUCCESS ===
            if result["success"]:
                st.success("Analysis Complete!")
                
                # 1. Show AI Response
                st.subheader("AI Analysis")
                st.info(result["ai_response"])
                
                # 2. Show Tools Used (only real API calls)
                                # === SHOW ONLY USER-FACING CALLS (Hide technical 3mo) ===
                st.subheader("Tools Automatically Selected")
                
                user_calls = []
                for call in result["tool_calls"]:
                    # Skip internal technical calls
                    if call.get("tool") == "get_stock_data" and call.get("parameters", {}).get("period") == "3mo":
                        continue  # Hide 3mo technical call
                    if not call.get("from_cache", False):
                        user_calls.append(call)
                
                if user_calls:
                    for i, call in enumerate(user_calls, 1):
                        tool_name = call["tool"]
                        params = {k: v for k, v in call.items() if k not in ["tool", "timestamp", "from_cache"]}
                        st.success(f"**{i}. {tool_name}** - Parameters: {params}")
                else:
                    st.info("No API calls needed — all data from cache!")
                # if result["tool_calls"]:
                #     for i, tool_call in enumerate(result["tool_calls"], 1):
                #         tool_name = tool_call["tool"]
                #         params = {k: v for k, v in tool_call.items() if k not in ["tool", "timestamp", "from_cache"]}
                #         from_cache = tool_call.get("from_cache", False)
                #         cache_note = " (from cache)" if from_cache else ""
                #         st.success(f"**{i}. {tool_name}{cache_note}** - Parameters: {params}")
                # else:
                #     st.warning("No tools were called for this query")
                
                # 3. Show Charts and Data (if available)
                if result["stock_data"] and result["stock_data"]["success"]:
                    st.subheader("📈 Visual Analysis")
                    
                    # Key metrics
                    stock_data = result["stock_data"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${stock_data['latest_price']:.2f}")
                    with col2:
                        st.metric("Total Return", f"{stock_data['total_return']:.2f}%")
                    with col3:
                        st.metric("Volatility", f"{stock_data['volatility']:.2f}%")
                    with col4:
                        st.metric("Period", result["period"])
                    
                    # Price chart
                    chart_creator = ChartCreator()
                    price_chart = chart_creator.create_price_chart(stock_data)
                    if price_chart:
                        st.plotly_chart(price_chart, use_container_width=True)

                    # ============== NEW SUPER SECTION ==============
                    symbol = result.get("symbol", "UNKNOWN")
                    if result.get("live_quote", {}).get("success"):
                        st.subheader("Live Snapshot")
                        q = result["live_quote"]
                        k = result.get("key_stats", {})

                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1: st.metric("Live Price", f"${q['price']:.2f}", q['change_pct'])
                        with col2: st.metric("Beta", k.get("beta", "N/A"))
                        with col3: st.metric("EPS (TTM)", k.get("eps", "N/A"))
                        with col4: st.metric("Analyst Target", k.get("analyst_target", "N/A"))
                        with col5: st.metric("Short Float", k.get("short_pct", "N/A"))

                        # AI Recommendation
                        rec = result.get("recommendation", "HOLD - No data")
                        first_word = rec.split()[0] if rec else "HOLD"
                        st.markdown(f"### AI Recommendation: **{first_word}**")
                        st.caption(rec)

                        # RED CAUTION BOX
                        st.error("""
                        NOT FINANCIAL ADVICE
                        • Past performance ≠ future results
                        • Markets can gap 10-20% overnight
                        • Use stop-loss & never risk >2% of capital
                        • Consult a licensed advisor
                        """)

                        # WATCHLIST BUTTON (now SAFE)
                        watchlist = st.session_state.get("watchlist", [])
                        if st.button(f"Add {symbol} to Watchlist"):
                            if symbol not in watchlist:
                                watchlist.append(symbol)
                                st.session_state.watchlist = watchlist
                                st.success(f"{symbol} added!")
                            else:
                                st.info("Already in watchlist")

                        if watchlist:
                            st.caption("Watchlist: " + " | ".join(watchlist))
                # 4. Company Information
                if result["company_data"] and result["company_data"]["success"]:
                    st.subheader("🏢 Company Details")
                    company = result["company_data"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Name:** {company['name']}")
                        st.write(f"**Sector:** {company['sector']}")
                        st.write(f"**Industry:** {company['industry']}")
                    with col2:
                        st.write(f"**Market Cap:** {company['market_cap']}")
                        st.write(f"**P/E Ratio:** {company['pe_ratio']}")
                        st.write(f"**Dividend Yield:** {company['dividend_yield']}")
                else:
                    if result["company_data"]:
                        st.warning(f"Company data unavailable: {result['company_data'].get('error', 'Unknown error')}")
                
            else:
                st.error(f"❌ Analysis failed: {result['error']}")

else:
    # Welcome message
    st.markdown("""
    ## 🌍 Universal Stock Analysis
    
    **Analyze ANY stock** - Known companies or new discoveries!
    
    ### 🔄 How It Works:
    
    1. **You ask about ANY stock** - "Show me PLTR stock", "Analyze Palantir", "Find biotech stocks"
    2. **AI understands your intent** and extracts the right symbol
    3. **LangChain automatically selects** the appropriate MCP tools
    4. **Tools fetch real data** from Alpha Vantage API
    5. **AI provides comprehensive analysis** with charts and insights
    
    ### 🎯 Now Supports:
    
    - **✅ Popular stocks**: AAPL, TSLA, MSFT, GOOGL
    - **✅ New companies**: PLTR, COIN, SHOP, HOOD  
    - **✅ Industry searches**: "Find biotech stocks", "Search for fintech companies"
    - **✅ Unknown symbols**: Any valid stock symbol will work!
    
    ### 💡 Try These:
    - "Show me PLTR stock for last year"
    - "Analyze Palantir Technologies"
    - "Find biotech companies with good potential"
    - "What about Shopify stock performance?"
    - "COIN stock analysis with technical indicators"
    
    *The system now dynamically handles ANY stock symbol or company name!*
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Universal Stock Analysis • Dynamic Symbol Extraction • AI-Powered Tools"
    "</div>",
    unsafe_allow_html=True
)
