import os
import csv
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# =====================
# BTC PRICE - FREE APIs (No key needed)
# =====================
def get_btc_price():
    """Free public APIs se BTC price fetch karo"""
    apis = [
        ("CoinGecko", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"),
        ("Binance", "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"),
        ("CoinCap", "https://api.coincap.io/v2/assets/bitcoin"),
    ]
    for name, url in apis:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if name == "CoinGecko":
                    price = data["bitcoin"]["usd"]
                    change = data["bitcoin"].get("usd_24h_change", 0)
                    return price, round(change, 2), name
                elif name == "Binance":
                    price = float(data["lastPrice"])
                    change = float(data["priceChangePercent"])
                    return price, change, name
                elif name == "CoinCap":
                    price = float(data["data"]["priceUsd"])
                    change = float(data["data"].get("changePercent24Hr", 0))
                    return price, change, name
        except:
            continue
    return None, None, None

# =====================
# BTC OHLCV DATA - FREE
# =====================
def get_btc_ohlcv(limit=100):
    """Binance public API se OHLCV - no key needed"""
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit={limit}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data, columns=[
                'timestamp','open','high','low','close','volume',
                'close_time','quote_vol','trades','taker_buy_base',
                'taker_buy_quote','ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)
            return df[['timestamp','open','high','low','close','volume']]
    except:
        pass
    return None

# =====================
# PATTERN ANALYSIS
# =====================
def analyze_btc_pattern(df):
    """Simple pattern analysis without external libraries"""
    if df is None or len(df) < 20:
        return "Data insufficient for analysis"

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values

    current = closes[-1]
    prev = closes[-2]

    # Trend
    ma7 = closes[-7:].mean()
    ma20 = closes[-20:].mean()
    ma50 = closes[-50:].mean() if len(closes) >= 50 else ma20

    if ma7 > ma20 > ma50:
        trend = "📈 Strong Uptrend"
    elif ma7 > ma20:
        trend = "📈 Uptrend"
    elif ma7 < ma20 < ma50:
        trend = "📉 Strong Downtrend"
    elif ma7 < ma20:
        trend = "📉 Downtrend"
    else:
        trend = "➡️ Sideways"

    # Support & Resistance
    support = min(lows[-20:])
    resistance = max(highs[-20:])

    # RSI (simple)
    gains = [max(closes[i]-closes[i-1], 0) for i in range(1, 15)]
    losses = [max(closes[i-1]-closes[i], 0) for i in range(1, 15)]
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    rsi_signal = "Overbought ⚠️" if rsi > 70 else ("Oversold ✅" if rsi < 30 else "Neutral")

    # Volume
    avg_vol = volumes[-10:].mean()
    curr_vol = volumes[-1]
    vol_signal = "High 🔥" if curr_vol > avg_vol * 1.5 else ("Low 😴" if curr_vol < avg_vol * 0.5 else "Normal")

    # Candle pattern
    body = closes[-1] - df['open'].values[-1]
    candle = "Bullish 🟢" if body > 0 else "Bearish 🔴"

    # Signal
    if trend in ["📈 Strong Uptrend", "📈 Uptrend"] and rsi < 70:
        signal = "🟢 BUY Setup"
        setup = "Uptrend mein buying opportunity"
    elif trend in ["📉 Strong Downtrend", "📉 Downtrend"] and rsi > 30:
        signal = "🔴 SELL Setup"
        setup = "Downtrend mein selling opportunity"
    else:
        signal = "⏳ WAIT"
        setup = "Clear setup nahi hai, wait karo"

    return f"""📊 *BTC ANALYSIS*

💰 Price: ${current:,.0f}
📈 Trend: {trend}
📡 Signal: {signal}

📐 *Levels:*
• Support: ${support:,.0f}
• Resistance: ${resistance:,.0f}

📊 *Indicators:*
• RSI(14): {rsi:.1f} — {rsi_signal}
• MA7: ${ma7:,.0f}
• MA20: ${ma20:,.0f}
• Volume: {vol_signal}
• Candle: {candle}

💡 *Setup:* {setup}"""

# =====================
# TRADE LOG READER
# =====================
def read_trades(log_file, n=10, agent_name="Agent"):
    """CSV se last N trades padho"""
    if not os.path.exists(log_file):
        return f"❌ {agent_name} ka koi trade log nahi mila!"

    try:
        df = pd.read_csv(log_file)
        # Sirf actual trades — HOLD nahi
        actual = df[df['action'].isin(['LONG', 'SHORT', 'LONG CLOSE', 'SHORT CLOSE', 'CLOSE', 'OPEN'])]
        if len(actual) == 0:
            actual = df[~df['action'].str.contains('HOLD', na=False)]

        if len(actual) == 0:
            return f"❌ {agent_name} ke koi trades nahi hain abhi tak!"

        last_n = actual.tail(n)
        total_trades = len(actual[actual['result'].isin(['WIN', 'LOSS'])])
        wins = len(actual[actual['result'] == 'WIN'])
        losses = len(actual[actual['result'] == 'LOSS'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        # Latest balance
        latest_balance = df['balance'].iloc[-1] if 'balance' in df.columns else 10000

        msg = f"🤖 *{agent_name} — Last {min(n, len(last_n))} Trades*\n\n"
        for _, row in last_n.iterrows():
            emoji = "✅" if row['result'] == 'WIN' else ("❌" if row['result'] == 'LOSS' else "🔵")
            profit = f"{row['profit_pct']:+.2f}%" if row['profit_pct'] != 0 else "OPEN"
            msg += f"{emoji} `{row['action']}` @ ${float(row['price']):,.0f} | {profit}\n"

        msg += f"\n📊 *Summary:*\n"
        msg += f"• Total Trades: {total_trades}\n"
        msg += f"• Wins: {wins} | Losses: {losses}\n"
        msg += f"• Win Rate: {win_rate:.1f}%\n"
        msg += f"• Balance: ${float(latest_balance):,.2f}"

        return msg
    except Exception as e:
        return f"❌ Error reading {agent_name} log: {str(e)}"

def get_agent_performance(log_file, agent_name):
    """Agent ka full performance summary"""
    if not os.path.exists(log_file):
        return f"❌ {agent_name} ka log nahi mila!"

    try:
        df = pd.read_csv(log_file)
        closed = df[df['result'].isin(['WIN', 'LOSS'])]

        if len(closed) == 0:
            return f"📊 {agent_name}: Koi closed trade nahi abhi tak"

        wins = len(closed[closed['result'] == 'WIN'])
        losses = len(closed[closed['result'] == 'LOSS'])
        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        profits = closed[closed['result'] == 'WIN']['profit_pct']
        loss_vals = closed[closed['result'] == 'LOSS']['profit_pct']

        best = profits.max() if len(profits) > 0 else 0
        worst = loss_vals.min() if len(loss_vals) > 0 else 0
        avg_profit = profits.mean() if len(profits) > 0 else 0
        avg_loss = loss_vals.mean() if len(loss_vals) > 0 else 0

        latest_balance = df['balance'].iloc[-1]
        initial_balance = 10000
        total_pnl = float(latest_balance) - initial_balance
        pnl_pct = (total_pnl / initial_balance) * 100

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        return f"""📊 *{agent_name} Performance*

💰 Balance: ${float(latest_balance):,.2f} {pnl_emoji}
📊 Total P&L: ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)

🎯 *Trade Stats:*
• Total Trades: {total}
• Wins: {wins} ✅ | Losses: {losses} ❌
• Win Rate: {win_rate:.1f}%

📈 *Best/Worst:*
• Best Trade: +{best:.2f}%
• Worst Trade: {worst:.2f}%
• Avg Win: +{avg_profit:.2f}%
• Avg Loss: {avg_loss:.2f}%"""

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =====================
# CLAUDE AI BRAIN
# =====================
def ask_claude(user_message, context=""):
    """Groq AI se jawab lo — FREE!"""
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY missing! GitHub Secrets mein add karo."

    system = """Tu ek expert crypto trading assistant hai jo Hinglish mein baat karta hai.
Tere paas 2 trading agents hain:
- Agent 1 (BTC RL Agent): Price action se seekhta hai, log file: trades_log.csv
- Agent 2 (SP Agent): Video scripts/knowledge base se seekhta hai, log file: sp_trades_log.csv

Tu concise aur helpful responses deta hai. Numbers aur facts clearly batata hai."""

    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}"})
        messages.append({"role": "assistant", "content": "Samajh gaya, context dekh liya."})
    messages.append({"role": "user", "content": user_message})

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            },
            timeout=30
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq error: {r.status_code} — {r.text[:100]}"
    except Exception as e:
        return f"Groq error: {str(e)}"

# =====================
# MESSAGE HANDLER
# =====================
def handle_message(text):
    """User ka message process karo"""
    text_lower = text.lower().strip()

    # BTC Price
    if any(w in text_lower for w in ['btc price', 'price', 'kitna hai', 'rate', 'btc kya chal']):
        price, change, source = get_btc_price()
        if price:
            emoji = "📈" if change and change > 0 else "📉"
            return f"""💰 *BTC Price*

${price:,.2f} {emoji}
24h Change: {change:+.2f}%
Source: {source}
Time: {datetime.now(IST).strftime('%H:%M IST')}"""
        return "❌ BTC price fetch nahi ho paya, thodi der mein try karo"

    # BTC Analysis / Pattern
    elif any(w in text_lower for w in ['analysis', 'pattern', 'setup', 'trend', 'signal', 'kya ban raha']):
        df = get_btc_ohlcv()
        return analyze_btc_pattern(df)

    # Agent 1 trades — IMPROVE pehle check hoga
    elif any(w in text_lower for w in ['agent 1', 'btc rl', 'pehla agent']) and not any(w in text_lower for w in ['improve', 'better', 'fix', 'theek', 'sudharo']):
        if any(w in text_lower for w in ['performance', 'summary', 'overall']):
            return get_agent_performance('trades_log.csv', 'Agent 1 (BTC RL)')
        n = 10
        for w in text_lower.split():
            if w.isdigit():
                n = int(w)
                break
        return read_trades('trades_log.csv', n, 'Agent 1 (BTC RL)')

    # Agent 2 trades — IMPROVE pehle check hoga
    elif any(w in text_lower for w in ['agent 2', 'sp agent', 'dusra agent', 'subhashis']) and not any(w in text_lower for w in ['improve', 'better', 'fix', 'theek', 'sudharo']):
        if any(w in text_lower for w in ['performance', 'summary', 'overall']):
            return get_agent_performance('sp_trades_log.csv', 'Agent 2 (SP Agent)')
        n = 10
        for w in text_lower.split():
            if w.isdigit():
                n = int(w)
                break
        return read_trades('sp_trades_log.csv', n, 'Agent 2 (SP Agent)')

    # Compare both agents
    elif any(w in text_lower for w in ['compare', 'dono', 'both agent', 'kaun better', 'kaun accha']):
        a1 = get_agent_performance('trades_log.csv', 'Agent 1')
        a2 = get_agent_performance('sp_trades_log.csv', 'Agent 2')
        return f"{a1}\n\n{'='*30}\n\n{a2}"

    # Last N trades (general)
    elif any(w in text_lower for w in ['last', 'trades', 'trade']):
        n = 10
        for w in text_lower.split():
            if w.isdigit():
                n = int(w)
                break
        if 'agent 2' in text_lower or 'sp' in text_lower:
            return read_trades('sp_trades_log.csv', n, 'Agent 2 (SP Agent)')
        return read_trades('trades_log.csv', n, 'Agent 1 (BTC RL)')

    # Help
    elif any(w in text_lower for w in ['help', 'kya kar', 'commands', 'menu']):
        return """🤖 *Main kya kar sakta hoon:*

💰 *Prices & Market:*
• `btc price` — Live BTC price
• `btc analysis` — Pattern & signals
• `btc trend` — Market setup

📊 *Agent 1 (BTC RL):*
• `agent 1 trades` — Last 10 trades
• `agent 1 last 5` — Last 5 trades
• `agent 1 performance` — Full summary

📊 *Agent 2 (SP Agent):*
• `agent 2 trades` — Last 10 trades
• `agent 2 performance` — Full summary

🔍 *Compare:*
• `compare agents` — Dono ka comparison

💬 *Aur bhi kuch poocho — main samjhunga!*"""

    # IMPROVE request
    elif any(w in text_lower for w in ['improve', 'better kar', 'fix kar', 'theek kar', 'improve karo', 'sudharo']):
        if 'agent 1' in text_lower or 'btc rl' in text_lower or 'pehla' in text_lower:
            perf = get_agent_performance('trades_log.csv', 'Agent 1 (BTC RL)')
            suggestions = """
🛠 *Agent 1 Improve Karne Ke Suggestions:*

1 Timesteps badhao: model.learn(total_timesteps=100000)
2 Stop Loss 5% se 3% karo
3 Cooldown 3 se 2 karo
4 Data limit 500 se 1000 karo
5 Reward function tune karo"""
            return perf + suggestions

        elif 'agent 2' in text_lower or 'sp' in text_lower or 'dusra' in text_lower:
            perf = get_agent_performance('sp_trades_log.csv', 'Agent 2 (SP Agent)')
            suggestions = """
🛠 *Agent 2 Improve Karne Ke Suggestions:*

1 Knowledge base update karo
2 Timesteps badhao: model.learn(total_timesteps=100000)
3 Market structure weight badhao
4 RR ratio 1:3 se 1:4 karo
5 Win rate 50% se upar lao"""
            return perf + suggestions

        else:
            a1 = get_agent_performance('trades_log.csv', 'Agent 1')
            a2 = get_agent_performance('sp_trades_log.csv', 'Agent 2')
            return a1 + chr(10)*2 + '='*20 + chr(10)*2 + a2 + chr(10)*2 + 'Kaunsa improve karna hai? agent 1 ya agent 2 bolو'

    # Claude AI se handle karo

        # Gather context
        context_parts = []
        price, change, _ = get_btc_price()
        if price:
            context_parts.append(f"Current BTC Price: ${price:,.2f} ({change:+.2f}%)")

        # Agent 1 quick stats
        try:
            df1 = pd.read_csv('trades_log.csv')
            closed1 = df1[df1['result'].isin(['WIN', 'LOSS'])]
            w1 = len(closed1[closed1['result'] == 'WIN'])
            t1 = len(closed1)
            b1 = df1['balance'].iloc[-1]
            context_parts.append(f"Agent 1: {t1} trades, {w1/t1*100:.1f}% win rate, Balance: ${float(b1):,.2f}")
        except:
            pass

        # Agent 2 quick stats
        try:
            df2 = pd.read_csv('sp_trades_log.csv')
            closed2 = df2[df2['result'].isin(['WIN', 'LOSS'])]
            w2 = len(closed2[closed2['result'] == 'WIN'])
            t2 = len(closed2)
            b2 = df2['balance'].iloc[-1]
            context_parts.append(f"Agent 2: {t2} trades, {w2/t2*100:.1f}% win rate, Balance: ${float(b2):,.2f}")
        except:
            pass

        context = "\n".join(context_parts)
        return ask_claude(text, context)

# =====================
# TELEGRAM SENDER
# =====================
def send_message(text, parse_mode="Markdown"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[BOT] {text}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    # Split long messages
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        try:
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk,
                "parse_mode": parse_mode
            }, timeout=10)
        except:
            # Try without markdown if fails
            try:
                requests.post(url, json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": chunk
                }, timeout=10)
            except:
                pass

def get_updates(offset=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": 30, "limit": 10}
    if offset:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, timeout=35)
        if r.status_code == 200:
            return r.json().get("result", [])
    except:
        pass
    return []

# =====================
# MAIN POLLING LOOP
# =====================
def main():
    print(f"🤖 Bot started! {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
    send_message("🤖 *Trading Assistant Online!*\n\n`help` likhkar commands dekho")

    offset = None
    while True:
        try:
            updates = get_updates(offset)
            for update in updates:
                offset = update["update_id"] + 1
                if "message" in update and "text" in update["message"]:
                    msg = update["message"]["text"]
                    chat_id = update["message"]["chat"]["id"]

                    # Only respond to authorized chat
                    if TELEGRAM_CHAT_ID and str(chat_id) != str(TELEGRAM_CHAT_ID):
                        continue

                    print(f"[MSG] {msg}")
                    response = handle_message(msg)
                    send_message(response)

        except KeyboardInterrupt:
            print("Bot stopped!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import time
            time.sleep(5)

if __name__ == "__main__":
    main()
