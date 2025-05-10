import requests
import json
import time
import logging
import random
import numpy as np
import pandas as pd
import telebot
import threading
import os
import datetime
import ccxt
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# Main entry point for the bot
def main():
    """Main entry point for the Pocket Option Trading Signal Bot"""
    logger.info("Starting Pocket Option Trading Signal Bot...")
    
    # Initialize Telegram bot with provided credentials
    telegram_bot = TelegramBot(TELEGRAM_TOKEN, CHAT_ID)
    
    # Start the bot
    try:
        telegram_bot.start_bot()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {str(e)}")
    
if __name__ == "__main__":
    main()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_signal_bot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TelegramBot:
    """Handles Telegram bot interactions"""
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.bot = telebot.TeleBot(token)
        self.signal_tracker = SignalTracker()
        self.running = False
        self.setup_commands()
        
    def setup_commands(self):
        """Set up bot commands"""
        # Handle /start command
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            self.running = True
            self.bot.reply_to(message, "🚀 Pocket Option Signal Bot started! You will now receive trading signals.")
            self.send_message("🤖 Bot is now active and monitoring markets for signals...")
            # Start signal generation in a separate thread
            threading.Thread(target=self.run_signal_generation).start()
            
        # Handle /stop command
        @self.bot.message_handler(commands=['stop'])
        def handle_stop(message):
            self.running = False
            self.bot.reply_to(message, "🛑 Pocket Option Signal Bot stopped. Type /start to resume signals.")
            
        # Handle /status command
        @self.bot.message_handler(commands=['status'])
        def handle_status(message):
            active_signals = self.signal_tracker.get_active_signals()
            win_rate = self.signal_tracker.get_win_rate()
            
            status_message = f"📊 *Bot Status:* {'🟢 Running' if self.running else '🔴 Stopped'}\n"
            status_message += f"🎯 *Current Win Rate:* {win_rate:.2f}%\n"
            status_message += f"📝 *Active Signals:* {len(active_signals)}\n\n"
            
            if active_signals:
                status_message += "*Current Active Signals:*\n"
                for signal_id, signal in active_signals.items():
                    expiry_time = signal['expiry_time'].strftime('%H:%M:%S')
                    status_message += f"- {signal['asset']} {signal['type']} (Expires at {expiry_time})\n"
            
            self.bot.reply_to(message, status_message, parse_mode="Markdown")
            
        # Handle /stats command
        @self.bot.message_handler(commands=['stats'])
        def handle_stats(message):
            completed_signals = self.signal_tracker.get_completed_signals(10)
            win_rate = self.signal_tracker.get_win_rate()
            
            stats_message = f"📈 *Performance Statistics:*\n"
            stats_message += f"🎯 *Overall Win Rate:* {win_rate:.2f}%\n\n"
            
            if completed_signals:
                stats_message += "*Recent Signal Results:*\n"
                for signal in completed_signals:
                    outcome_emoji = "✅" if signal['outcome'] == 'WIN' else "❌"
                    exit_time = signal['exit_time'].strftime('%H:%M:%S')
                    stats_message += f"{outcome_emoji} {signal['asset']} {signal['type']} - {signal['reason']} ({exit_time})\n"
            
            self.bot.reply_to(message, stats_message, parse_mode="Markdown")
        
        # Handle text messages (including "start" and "stop")
        @self.bot.message_handler(func=lambda message: True)
        def handle_text(message):
            if message.text.lower() == "start":
                self.running = True
                self.bot.reply_to(message, "🚀 Pocket Option Signal Bot started! You will now receive trading signals.")
                self.send_message("🤖 Bot is now active and monitoring markets for signals...")
                # Start signal generation in a separate thread
                threading.Thread(target=self.run_signal_generation).start()
            
            elif message.text.lower() == "stop":
                self.running = False
                self.bot.reply_to(message, "🛑 Pocket Option Signal Bot stopped. Type 'start' to resume signals.")
            
            else:
                help_text = (
                    "💬 *Available Commands:*\n"
                    "• start - Start receiving trading signals\n"
                    "• stop - Stop receiving trading signals\n"
                    "• /status - Check bot status and active signals\n"
                    "• /stats - View performance statistics"
                )
                self.bot.reply_to(message, help_text, parse_mode="Markdown")
    
    def send_message(self, message, parse_mode="Markdown"):
        """Send a message to the configured chat ID"""
        try:
            self.bot.send_message(self.chat_id, message, parse_mode=parse_mode)
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
            return False
    
    def send_signal(self, signal, asset, expiry_minutes):
        """Format and send a signal to Telegram"""
        signal_id = f"{asset.replace('/', '')}_{int(time.time())}"
        
        # Track the signal
        tracked_signal = self.signal_tracker.add_signal(signal_id, signal, asset, expiry_minutes)
        
        # Format signal message
        expiry_time = tracked_signal['expiry_time'].strftime('%H:%M:%S')
        entry_time = tracked_signal['entry_time'].strftime('%H:%M:%S')
        
        signal_type = tracked_signal['type']
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        
        message = (
            f"{emoji} *{signal_type} SIGNAL DETECTED*\n\n"
            f"🏛️ *Asset:* {asset}\n"
            f"📊 *Strategy:* {tracked_signal['reason']}\n"
            f"⏱️ *Expiry Time:* {expiry_minutes} minutes\n"
            f"🕒 *Entry Time:* {entry_time}\n"
            f"⌛ *Expiry At:* {expiry_time}\n"
            f"💰 *Entry Price:* {tracked_signal['entry_price']:.5f}\n"
            f"🎯 *Signal Strength:* {tracked_signal['strength']*100:.1f}%\n\n"
            f"⚠️ *Risk Warning:* Always use proper risk management."
        )
        
        # Send signal to Telegram
        self.send_message(message)
        
        return signal_id
    
    def send_outcome(self, signal):
        """Send the outcome of a completed signal"""
        asset = signal['asset']
        signal_type = signal['type']
        outcome = signal['outcome']
        
        emoji = "✅" if outcome == "WIN" else "❌"
        type_emoji = "🟢" if signal_type == "BUY" else "🔴"
        
        profit_loss = abs(signal['exit_price'] - signal['entry_price'])
        profit_loss_pct = (profit_loss / signal['entry_price']) * 100
        
        message = (
            f"{emoji} *SIGNAL OUTCOME: {outcome}*\n\n"
            f"🏛️ *Asset:* {asset}\n"
            f"{type_emoji} *Direction:* {signal_type}\n"
            f"📊 *Strategy:* {signal['reason']}\n"
            f"💰 *Entry Price:* {signal['entry_price']:.5f}\n"
            f"💸 *Exit Price:* {signal['exit_price']:.5f}\n"
            f"📈 *Change:* {profit_loss_pct:.2f}%\n"
            f"⏱️ *Signal Duration:* {signal['expiry_minutes']} minutes\n\n"
        )
        
        # Get current win rate
        win_rate = self.signal_tracker.get_win_rate()
        message += f"📝 *Current Win Rate:* {win_rate:.2f}%"
        
        # Send outcome to Telegram
        self.send_message(message)
    
    def run_signal_generation(self):
        """Run the signal generation loop"""
        data_provider = PocketOptionDataProvider()
        
        while self.running:
            try:
                # Check for completed signals
                completed_signal_ids = self.signal_tracker.check_outcomes()
                
                # Send outcome messages for completed signals
                for signal_id in completed_signal_ids:
                    completed_signals = self.signal_tracker.get_completed_signals()
                    for signal in completed_signals:
                        if signal['id'] == signal_id:
                            self.send_outcome(signal)
                            break
                
                # Generate new signals
                for asset in ASSETS:
                    if not self.running:
                        break
                        
                    try:
                        # Get market data
                        df = data_provider.get_market_data(asset, timeframe=5, limit=100)
                        
                        # Process for signals
                        if not df.empty:
                            df = TradingStrategies.add_indicators(df)
                            signals = TradingStrategies.generate_signals(df)
                            
                            # Get only the most recent signal with highest strength
                            if signals:
                                # Sort by strength
                                signals.sort(key=lambda x: x['strength'], reverse=True)
                                best_signal = signals[0]
                                
                                # Only send if we don't have an active signal for this asset
                                asset_has_active_signal = any(
                                    s['asset'] == asset and s['status'] == 'active' 
                                    for s in self.signal_tracker.get_active_signals().values()
                                )
                                
                                if not asset_has_active_signal and best_signal['strength'] >= 0.85:
                                    # Select a random expiry time between 1-5 minutes
                                    expiry_minutes = random.choice(TIMEFRAMES)
                                    
                                    # Send the signal
                                    self.send_signal(best_signal, asset, expiry_minutes)
                                    
                                    # Limit to prevent too many signals
                                    time.sleep(30)
                    
                    except Exception as e:
                        logger.error(f"Error processing asset {asset}: {str(e)}")
                
                # Wait before next scan cycle
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
    
    def start_bot(self):
        """Start the bot polling for messages"""
        logger.info("Starting Telegram bot...")
        
        try:
            self.bot.polling(none_stop=True, timeout=60)
        except Exception as e:
            logger.error(f"Bot polling error: {str(e)}")
            # Try to restart polling
            time.sleep(10)
            self.start_bot()

# Telegram Bot Configuration
TELEGRAM_TOKEN = "8032235672:AAGnB_MG7K96AY5FsZXuZPqVzDZjlCWNhgUBot"
CHAT_ID = "6210098588"

# Trading Configuration
ASSETS = [
    "EUR/USD", "GBP/USD", "AUD/USD", "USD/CAD", "USD/JPY", 
    "EUR/JPY", "GBP/JPY", "BTC/USD", "ETH/USD"
]
TIMEFRAMES = [1, 3, 5]  # Expiry times in minutes
STRATEGIES = [
    "Smart Money Concept", 
    "Liquidity Grab", 
    "Quasimodo Pattern", 
    "Support and Resistance", 
    "Sniper Entry"
]

# Bot state
is_running = False
active_signals = {}  # To store active signals and track outcomes

class PocketOptionDataProvider:
    """
    Handles data retrieval from market sources.
    In a real implementation, this would connect directly to Pocket Option.
    For this example, we'll use a combination of public APIs and simulation.
    """
    def __init__(self):
        # Use a public API for actual price data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
    def convert_symbol(self, pocket_option_symbol):
        """Convert Pocket Option symbol format to a format suitable for the exchange"""
        if '/' in pocket_option_symbol:
            base, quote = pocket_option_symbol.split('/')
            if base == 'BTC' or base == 'ETH':
                return f"{base}/{quote}"
            else:
                return f"{base}{quote}"
        return pocket_option_symbol
    
    def get_market_data(self, asset, timeframe=1, limit=100):
        """
        Fetch market data for the given asset
        
        Args:
            asset: Market symbol (e.g., "EUR/USD")
            timeframe: Candle timeframe in minutes
            limit: Number of candles to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            symbol = self.convert_symbol(asset)
            
            # Map minutes to exchange timeframe format
            tf_mapping = {1: '1m', 3: '3m', 5: '5m', 15: '15m', 30: '30m', 60: '1h'}
            exchange_tf = tf_mapping.get(timeframe, '5m')
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, exchange_tf, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {asset}: {str(e)}")
            
            # Fallback to simulated data if API fails
            return self.generate_simulated_data(limit)
    
    def generate_simulated_data(self, length=100):
        """Generate simulated price data if API fails"""
        base_price = random.uniform(1.0, 100.0)
        volatility = base_price * 0.002
        
        timestamps = [datetime.datetime.now() - datetime.timedelta(minutes=i) for i in range(length, 0, -1)]
        
        # Generate random walk prices
        closes = [base_price]
        for i in range(1, length):
            close = closes[-1] + random.uniform(-volatility, volatility)
            closes.append(close)
        
        # Generate OHLC from close prices
        data = []
        for i in range(length):
            close = closes[i]
            open_price = closes[i-1] if i > 0 else close * (1 - random.uniform(0, 0.001))
            high = max(open_price, close) * (1 + random.uniform(0, 0.001))
            low = min(open_price, close) * (1 - random.uniform(0, 0.001))
            volume = random.uniform(100, 1000)
            
            data.append([timestamps[i], open_price, high, low, close, volume])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df

class TradingStrategies:
    """Implements various trading strategies"""
    
    @staticmethod
    def add_indicators(df):
        """Add common technical indicators to the dataframe"""
        # Trend indicators
        df['sma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['close'])
        df['bollinger_mavg'] = bollinger.bollinger_mavg()
        df['bollinger_high'] = bollinger.bollinger_hband()
        df['bollinger_low'] = bollinger.bollinger_lband()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ATR for volatility
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        # Volume indicators
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        return df
    
    @staticmethod
    def smart_money_concept(df):
        """
        Smart Money Concept strategy
        Looks for institutional order blocks and liquidity
        """
        signals = []
        
        # Need at least 50 candles for this strategy
        if len(df) < 50:
            return signals
        
        # Use clean data for calculations
        df = df.copy().dropna()
        
        # Add some SMC-specific calculations
        df['body_size'] = abs(df['close'] - df['open'])
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        # Calculate 20-period high and low
        df['highest_20'] = df['high'].rolling(window=20).max()
        df['lowest_20'] = df['low'].rolling(window=20).min()
        
        # Identify potential imbalance areas (Fair Value Gaps)
        df['fvg_bullish'] = (df['low'].shift(2) > df['high'].shift(0)) & (df['is_bullish'].shift(1))
        df['fvg_bearish'] = (df['high'].shift(2) < df['low'].shift(0)) & (df['is_bearish'].shift(1))
        
        # Identify potential order blocks
        df['bullish_ob'] = (df['is_bearish']) & (df['close'].shift(-1) > df['high']) & (df['body_size'] > df['body_size'].rolling(10).mean())
        df['bearish_ob'] = (df['is_bullish']) & (df['close'].shift(-1) < df['low']) & (df['body_size'] > df['body_size'].rolling(10).mean())
        
        # Look for entries based on price returning to order blocks
        for i in range(5, len(df)-1):
            # Find most recent bullish order block
            recent_bullish_ob = False
            for j in range(i-1, max(i-20, 0), -1):
                if df.iloc[j]['bullish_ob']:
                    recent_bullish_ob = j
                    break
            
            # Find most recent bearish order block
            recent_bearish_ob = False
            for j in range(i-1, max(i-20, 0), -1):
                if df.iloc[j]['bearish_ob']:
                    recent_bearish_ob = j
                    break
            
            # Bullish signal: Price comes back to bullish order block
            if recent_bullish_ob and df.iloc[i]['low'] <= df.iloc[recent_bullish_ob]['high'] and df.iloc[i]['low'] >= df.iloc[recent_bullish_ob]['low']:
                if df.iloc[i]['close'] > df.iloc[i]['open'] and df.iloc[i]['rsi'] < 40:
                    signals.append({
                        'timestamp': df.iloc[i]['timestamp'],
                  
