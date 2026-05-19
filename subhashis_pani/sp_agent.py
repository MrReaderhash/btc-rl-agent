import ccxt
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os
import csv
import json
import requests
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# =====================
# TELEGRAM CONFIG
# =====================
TELEGRAM_TOKEN = "8644074664:AAH-pc-pp4FUpEKsyyUb5rEabiU7eDWfC2Q"
CHAT_ID        = "711544016"

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        if r.status_code != 200:
            print(f"   Telegram error: {r.text}")
    except Exception as e:
        print(f"   Telegram error: {e}")

def send_telegram_file(filepath, caption=""):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        with open(filepath, 'rb') as f:
            requests.post(url, data={"chat_id": CHAT_ID, "caption": caption},
                          files={"document": f}, timeout=15)
    except Exception as e:
        print(f"   Telegram file error: {e}")

def candle_time_ist(timestamp):
    return pd.Timestamp(timestamp).tz_localize('UTC').tz_convert(IST).strftime("%Y-%m-%d %H:%M")

# =====================
# KNOWLEDGE BASE
# =====================
def load_knowledge():
    if os.path.exists('knowledge_base.json'):
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            kb = json.load(f)
        print(f"Knowledge base loaded!")
        print(f"  Entry rules  : {len(kb.get('entry_rules', []))}")
        print(f"  Exit rules   : {len(kb.get('exit_rules', []))}")
        print(f"  Key concepts : {len(kb.get('key_concepts', []))}")
        return kb
    return None

# =====================
# MARKET STRUCTURE
# (purana code — same rakha)
# =====================
def get_market_structure(df, current_step):
    if current_step < 20:
        return 0
    highs = df['high'].iloc[current_step-20:current_step].values
    lows  = df['low'].iloc[current_step-20:current_step].values
    recent_highs = highs[-10:]
    recent_lows  = lows[-10:]
    older_highs  = highs[:10]
    older_lows   = lows[:10]
    if recent_highs.max() > older_highs.max() and recent_lows.min() > older_lows.min():
        return 1.0
    elif recent_highs.max() < older_highs.max() and recent_lows.min() < older_lows.min():
        return -1.0
    return 0.0

def find_swing_low(df, current_step, lookback=10):
    return df['low'].iloc[current_step-lookback:current_step].min()

def find_swing_high(df, current_step, lookback=10):
    return df['high'].iloc[current_step-lookback:current_step].max()

def get_candle_features(candles):
    features = []
    for c in candles:
        o, h, l, cl = c[0], c[1], c[2], c[3]
        body        = cl - o
        body_size   = abs(body) / (h - l + 1e-8)
        upper_wick  = (h - max(o, cl)) / (h - l + 1e-8)
        lower_wick  = (min(o, cl) - l) / (h - l + 1e-8)
        direction   = 1.0 if body > 0 else -1.0
        features.extend([body_size, upper_wick, lower_wick, direction])
    return features

# =====================
# OBSERVATION BUILDER
# Purana structure — old model ke saath compatible
# 181 features → pad to 185
# =====================
def build_obs(df, step, state):
    window       = df[['open', 'high', 'low', 'close', 'volume']].values
    candles_ohlc = df[['open', 'high', 'low', 'close']].values

    candles_20   = candles_ohlc[step-20:step]
    normalized   = (candles_20 - candles_20.mean(axis=0)) / (candles_20.std(axis=0) + 1e-8)
    ohlc_flat    = normalized.flatten()                                    # 80
    candle_feats = np.array(get_candle_features(candles_20), dtype=np.float32)  # 80

    current_price = candles_ohlc[step-1, 3]

    high_10 = candles_ohlc[step-10:step, 1].max()
    low_10  = candles_ohlc[step-10:step, 2].min()
    high_20 = candles_ohlc[step-20:step, 1].max()
    low_20  = candles_ohlc[step-20:step, 2].min()
    high_50 = candles_ohlc[step-50:step, 1].max()
    low_50  = candles_ohlc[step-50:step, 2].min()
    price_range = high_50 - low_50 + 1e-8

    breakout_feats = np.array([
        (current_price - high_10) / price_range,
        (current_price - low_10)  / price_range,
        (current_price - high_20) / price_range,
        (current_price - low_20)  / price_range,
        (current_price - high_50) / price_range,
        (current_price - low_50)  / price_range,
    ], dtype=np.float32)  # 6

    market_structure = get_market_structure(df, step)

    closes     = candles_ohlc[step-10:step, 3]
    mom_3      = (closes[-1] - closes[-3])  / (closes[-3]  + 1e-8)
    mom_5      = (closes[-1] - closes[-5])  / (closes[-5]  + 1e-8)
    mom_10     = (closes[-1] - closes[-10]) / (closes[-10] + 1e-8)
    volatility = closes.std() / (current_price + 1e-8)

    vol      = window[step-10:step, 4]
    vol_mean = vol.mean() + 1e-8
    win_rate = state['winning_trades'] / (state['total_trades'] + 1e-8)

    extra_feats = np.array([
        market_structure,
        mom_3, mom_5, mom_10,
        volatility,
        vol[-1] / vol_mean,
        vol[-3:].mean() / vol_mean,
        float(state['position']),
        float(state['cooldown']) / 5.0,
        float(state['hold_count']) / 20.0,
        win_rate,
    ], dtype=np.float32)  # 11

    # Purana trade_feats — sl_dist, tgt_dist, rr_ratio, current_pnl
    pos = state['position']
    sl  = state['stop_loss']
    tgt = state['target']
    if pos != 0 and sl > 0:
        sl_dist     = abs(current_price - sl)  / (current_price + 1e-8)
        tgt_dist    = abs(tgt - current_price) / (current_price + 1e-8)
        rr_ratio    = tgt_dist / (sl_dist + 1e-8)
        current_pnl = (current_price - state['entry_price']) / state['entry_price'] if pos == 1 \
                      else (state['entry_price'] - current_price) / state['entry_price']
    else:
        sl_dist = tgt_dist = rr_ratio = current_pnl = 0.0

    trade_feats = np.array([sl_dist, tgt_dist, rr_ratio, current_pnl], dtype=np.float32)  # 4

    # Total: 80+80+6+11+4 = 181 → pad to 185
    obs = np.concatenate([
        ohlc_flat, candle_feats, breakout_feats, extra_feats, trade_feats
    ]).astype(np.float32)

    if len(obs) < 185:
        obs = np.pad(obs, (0, 185 - len(obs)))
    return obs[:185]

# =====================
# STATE: SAVE / LOAD
# =====================
STATE_FILE = 'sp_state.json'

def load_state():
    today_ist = datetime.now(IST).strftime("%Y-%m-%d")
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        if state.get('last_date', '') != today_ist:
            print(f"   Naya din! Trade count reset.")
            state['total_trades'] = 0
            state['last_date']    = today_ist
        return state
    return {
        'balance':       10000.0,
        'position':      0,
        'entry_price':   0.0,
        'stop_loss':     0.0,
        'target':        0.0,
        'trailing_sl':   0.0,
        'hold_count':    0,
        'cooldown':      0,
        'total_trades':  0,
        'winning_trades':0,
        'last_date':     today_ist
    }

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# =====================
# LOG HELPER
# =====================
def log_trade(trade_dict):
    log_file    = 'sp_trades_log.csv'
    file_exists = os.path.exists(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'date', 'action', 'price', 'sl', 'target',
            'profit_pct', 'balance', 'result'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_dict)

# =====================
# TRAINING ENV
# Purana code — bilkul same
# =====================
class SPTradingEnv(gym.Env):
    def __init__(self, df, knowledge=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.knowledge = knowledge
        self.current_step = 60
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.target = 0.0
        self.trailing_sl = 0.0
        self.hold_count = 0
        self.cooldown = 0
        self.recent_trades = []
        self.total_trades = 0
        self.winning_trades = 0

        self.action_space = spaces.Discrete(4)   # Purana — old model compatible
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(185,), dtype=np.float32
        )

    def _get_obs(self):
        state = {
            'position':      self.position,
            'entry_price':   self.entry_price,
            'stop_loss':     self.stop_loss,
            'target':        self.target,
            'hold_count':    self.hold_count,
            'cooldown':      self.cooldown,
            'total_trades':  self.total_trades,
            'winning_trades':self.winning_trades,
        }
        return build_obs(self.df, self.current_step, state)

    def step(self, action):
        action        = int(action)
        current_price = self.df['close'].iloc[self.current_step]
        reward        = 0
        forced_close  = False

        if self.cooldown > 0:
            self.cooldown -= 1
            action = 0

        self.recent_trades.append(1 if action in [1, 2] else 0)
        if len(self.recent_trades) > 5:
            self.recent_trades.pop(0)
        overtrade  = sum(self.recent_trades) >= 3
        market_str = get_market_structure(self.df, self.current_step)

        if action == 1 and self.position == 0:
            if overtrade:
                reward = -0.05
            else:
                sl          = find_swing_low(self.df, self.current_step)
                sl_distance = current_price - sl
                trend_bonus = 0.01 if market_str == 1.0 else -0.01
                if sl_distance <= 0 or sl_distance > current_price * 0.05:
                    reward = -0.01
                else:
                    target = current_price + (sl_distance * 3)
                    self.position    = 1
                    self.entry_price = current_price
                    self.stop_loss   = sl
                    self.target      = target
                    self.trailing_sl = sl
                    self.hold_count  = 0
                    self.total_trades += 1
                    reward = trend_bonus

        elif action == 2 and self.position == 0:
            if overtrade:
                reward = -0.05
            else:
                sl          = find_swing_high(self.df, self.current_step)
                sl_distance = sl - current_price
                trend_bonus = 0.01 if market_str == -1.0 else -0.01
                if sl_distance <= 0 or sl_distance > current_price * 0.05:
                    reward = -0.01
                else:
                    target = current_price - (sl_distance * 3)
                    self.position    = -1
                    self.entry_price = current_price
                    self.stop_loss   = sl
                    self.target      = target
                    self.trailing_sl = sl
                    self.hold_count  = 0
                    self.total_trades += 1
                    reward = trend_bonus

        elif self.position != 0:
            self.hold_count += 1

            if self.position == 1:
                if current_price > self.entry_price:
                    new_trail = current_price - (self.entry_price - self.stop_loss)
                    self.trailing_sl = max(self.trailing_sl, new_trail)
                if current_price <= self.trailing_sl:
                    profit = (current_price - self.entry_price) / self.entry_price
                    reward = profit * 150
                    self.balance *= (1 + profit)
                    forced_close = True
                elif current_price >= self.target:
                    profit = (current_price - self.entry_price) / self.entry_price
                    reward = profit * 200
                    self.balance *= (1 + profit)
                    self.winning_trades += 1
                    forced_close = True
                elif action == 0 and current_price > self.entry_price:
                    reward = 0.005
                elif action == 0 and current_price < self.entry_price:
                    reward = -0.005

            elif self.position == -1:
                if current_price < self.entry_price:
                    new_trail = current_price + (self.stop_loss - self.entry_price)
                    self.trailing_sl = min(self.trailing_sl, new_trail)
                if current_price >= self.trailing_sl:
                    profit = (self.entry_price - current_price) / self.entry_price
                    reward = profit * 150
                    self.balance *= (1 + profit)
                    forced_close = True
                elif current_price <= self.target:
                    profit = (self.entry_price - current_price) / self.entry_price
                    reward = profit * 200
                    self.balance *= (1 + profit)
                    self.winning_trades += 1
                    forced_close = True
                elif action == 0 and current_price < self.entry_price:
                    reward = 0.005
                elif action == 0 and current_price > self.entry_price:
                    reward = -0.005

            if action == 3 and not forced_close:
                if self.position == 1:
                    profit = (current_price - self.entry_price) / self.entry_price
                else:
                    profit = (self.entry_price - current_price) / self.entry_price
                reward = profit * (80 if profit > 0 else 100)
                self.balance *= (1 + profit)
                if profit > 0:
                    self.winning_trades += 1
                forced_close = True

            if forced_close:
                self.position    = 0
                self.entry_price = 0.0
                self.stop_loss   = 0.0
                self.target      = 0.0
                self.trailing_sl = 0.0
                self.hold_count  = 0
                self.cooldown    = 3

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None):
        self.current_step = 60
        self.balance      = self.initial_balance
        self.position     = 0
        self.entry_price  = 0.0
        self.stop_loss    = 0.0
        self.target       = 0.0
        self.trailing_sl  = 0.0
        self.hold_count   = 0
        self.cooldown     = 0
        self.recent_trades = []
        self.total_trades  = 0
        self.winning_trades = 0
        return self._get_obs(), {}

# =====================
# MAIN
# =====================
now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
print(f"\n{'='*50}")
print(f"SUBHASHISH PANI RL AGENT v8")
print(f"Time (IST): {now_ist}")
print(f"{'='*50}")

send_telegram(f"🤖 SP Agent v8 Started\n⏰ {now_ist}")

knowledge = load_knowledge()

# ---- DATA FETCH ----
print("\n1. BTC data fetch ho raha hai...")
for attempt in range(5):
    try:
        exchange = ccxt.kraken()
        ohlcv    = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=200)
        print(f"   Data mila! Candles: {len(ohlcv)}")
        break
    except Exception as e:
        print(f"   Attempt {attempt+1} failed: {e}")
        if attempt == 4:
            send_telegram(f"❌ Data fetch failed!\n{e}")
            raise
        import time
        time.sleep(5)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.reset_index(drop=True)

data_file = 'sp_btc_data.csv'
if os.path.exists(data_file):
    old_df      = pd.read_csv(data_file)
    combined_df = pd.concat([old_df, df], ignore_index=True)
    combined_df = combined_df.drop_duplicates('timestamp').reset_index(drop=True)
    combined_df.to_csv(data_file, index=False)
    train_df = combined_df
else:
    df.to_csv(data_file, index=False)
    train_df = df

print(f"   Total candles: {len(train_df)}")

# ---- TRAINING ----
print("\n2. Training ho raha hai...")
train_env  = SPTradingEnv(train_df, knowledge)
model_file = 'sp_rl_model'

if os.path.exists(f'{model_file}.zip'):
    model = PPO.load(model_file, env=train_env)
    model.learn(total_timesteps=10000)   # Quick update
    print("   Model updated!")
else:
    model = PPO("MlpPolicy", train_env, verbose=0,
                n_steps=2048, batch_size=64)
    model.learn(total_timesteps=50000)
    print("   Naya model bana!")

model.save(model_file)
print("   Model saved!")

# =====================
# REAL-TIME LIVE TRADING
# Sirf latest candle pe decision
# =====================
print("\n3. Real-time decision...")

state = load_state()
print(f"   Balance     : ${state['balance']:,.2f}")
print(f"   Position    : {'LONG' if state['position']==1 else 'SHORT' if state['position']==-1 else 'NONE'}")
print(f"   Trades today: {state['total_trades']}/5")

step          = len(df) - 1   # Latest candle
current_price = df['close'].iloc[step]
current_time  = candle_time_ist(df['timestamp'].iloc[step])

print(f"   Latest candle: {current_time} @ ${current_price:,.1f}")

# Cooldown
if state['cooldown'] > 0:
    state['cooldown'] -= 1
    print(f"   Cooldown: {state['cooldown']} candles bacha")
    save_state(state)
    send_telegram(f"⏳ Cooldown active\n⏰ {current_time}\n💰 ${current_price:,.1f}")
    import sys; sys.exit(0)

# Observation + prediction
obs    = build_obs(df, step, state)
action = int(model.predict(obs.reshape(1, -1))[0])
print(f"   RL action: {['HOLD','LONG','SHORT','CLOSE'][action]}")

market_str = get_market_structure(df, step)
at_limit   = state['total_trades'] >= 5

# ---- EXISTING POSITION CHECK ----
if state['position'] != 0:
    state['hold_count'] += 1

    # Trailing SL update
    if state['position'] == 1 and current_price > state['entry_price']:
        new_trail = current_price - (state['entry_price'] - state['stop_loss'])
        state['trailing_sl'] = max(state['trailing_sl'], new_trail)
    elif state['position'] == -1 and current_price < state['entry_price']:
        new_trail = current_price + (state['stop_loss'] - state['entry_price'])
        state['trailing_sl'] = min(state['trailing_sl'], new_trail)

    closed  = False
    profit  = 0.0
    reason  = ''

    if state['position'] == 1:
        if current_price <= state['trailing_sl']:
            profit = (current_price - state['entry_price']) / state['entry_price']
            reason = 'Trail SL'
            closed = True
        elif current_price >= state['target']:
            profit = (current_price - state['entry_price']) / state['entry_price']
            reason = 'Target ✅'
            closed = True

    elif state['position'] == -1:
        if current_price >= state['trailing_sl']:
            profit = (state['entry_price'] - current_price) / state['entry_price']
            reason = 'Trail SL'
            closed = True
        elif current_price <= state['target']:
            profit = (state['entry_price'] - current_price) / state['entry_price']
            reason = 'Target ✅'
            closed = True

    if closed:
        state['balance'] *= (1 + profit)
        result = 'WIN' if profit > 0 else 'LOSS'
        if profit > 0:
            state['winning_trades'] += 1
        direction = 'LONG' if state['position'] == 1 else 'SHORT'

        print(f"   {'✅' if result=='WIN' else '❌'} {direction} CLOSE ({reason}) @ ${current_price:,.1f} — {result}")
        send_telegram(
            f"{'✅' if result=='WIN' else '❌'} {direction} CLOSE — {result} {'🎉' if result=='WIN' else '😞'}\n"
            f"⏰ Candle  : {current_time}\n"
            f"📌 Reason  : {reason}\n"
            f"💰 Price   : ${current_price:,.1f}\n"
            f"📈 P&L     : {profit*100:.2f}%\n"
            f"💼 Balance : ${state['balance']:,.2f}"
        )

        log_trade({
            'date': current_time,
            'action': f'{direction} CLOSE',
            'price': current_price,
            'sl': round(state['stop_loss'], 1),
            'target': round(state['target'], 1),
            'profit_pct': round(profit, 4),
            'balance': round(state['balance'], 2),
            'result': result
        })

        state['position']    = 0
        state['entry_price'] = 0.0
        state['stop_loss']   = 0.0
        state['target']      = 0.0
        state['trailing_sl'] = 0.0
        state['hold_count']  = 0
        state['cooldown']    = 3

    else:
        # Position open hai — update bhejo
        direction = 'LONG' if state['position'] == 1 else 'SHORT'
        pnl = (current_price - state['entry_price']) / state['entry_price'] if state['position'] == 1 \
              else (state['entry_price'] - current_price) / state['entry_price']
        print(f"   📊 {direction} open | Entry: ${state['entry_price']:,.1f} | Trail SL: ${state['trailing_sl']:,.1f} | PnL: {pnl*100:.2f}%")
        send_telegram(
            f"📊 {direction} Position Update\n"
            f"⏰ {current_time}\n"
            f"💰 Current : ${current_price:,.1f}\n"
            f"🎯 Target  : ${state['target']:,.1f}\n"
            f"🛑 Trail SL: ${state['trailing_sl']:,.1f}\n"
            f"📈 PnL     : {pnl*100:.2f}%\n"
            f"💼 Balance : ${state['balance']:,.2f}"
        )

# ---- NEW ENTRY ----
if state['position'] == 0 and not at_limit and state['cooldown'] == 0:

    if action == 1:
        sl          = find_swing_low(df, step)
        sl_distance = current_price - sl
        if sl_distance > 0 and sl_distance <= current_price * 0.05:
            target = current_price + (sl_distance * 3)
            state['position']    = 1
            state['entry_price'] = current_price
            state['stop_loss']   = sl
            state['target']      = target
            state['trailing_sl'] = sl
            state['hold_count']  = 0
            state['total_trades'] += 1

            print(f"   🟢 LONG @ ${current_price:,.1f} | SL: ${sl:,.1f} | Target: ${target:,.1f}")
            send_telegram(
                f"🟢 LONG ENTRY\n"
                f"⏰ Candle : {current_time}\n"
                f"💰 Price  : ${current_price:,.1f}\n"
                f"🛑 SL     : ${sl:,.1f}\n"
                f"🎯 Target : ${target:,.1f}\n"
                f"💼 Balance: ${state['balance']:,.2f}"
            )
            log_trade({
                'date': current_time, 'action': 'LONG',
                'price': current_price,
                'sl': round(sl, 1), 'target': round(target, 1),
                'profit_pct': 0,
                'balance': round(state['balance'], 2),
                'result': 'OPEN'
            })

    elif action == 2:
        sl          = find_swing_high(df, step)
        sl_distance = sl - current_price
        if sl_distance > 0 and sl_distance <= current_price * 0.05:
            target = current_price - (sl_distance * 3)
            state['position']    = -1
            state['entry_price'] = current_price
            state['stop_loss']   = sl
            state['target']      = target
            state['trailing_sl'] = sl
            state['hold_count']  = 0
            state['total_trades'] += 1

            print(f"   🔴 SHORT @ ${current_price:,.1f} | SL: ${sl:,.1f} | Target: ${target:,.1f}")
            send_telegram(
                f"🔴 SHORT ENTRY\n"
                f"⏰ Candle : {current_time}\n"
                f"💰 Price  : ${current_price:,.1f}\n"
                f"🛑 SL     : ${sl:,.1f}\n"
                f"🎯 Target : ${target:,.1f}\n"
                f"💼 Balance: ${state['balance']:,.2f}"
            )
            log_trade({
                'date': current_time, 'action': 'SHORT',
                'price': current_price,
                'sl': round(sl, 1), 'target': round(target, 1),
                'profit_pct': 0,
                'balance': round(state['balance'], 2),
                'result': 'OPEN'
            })

    else:
        print(f"   ⏸️  HOLD — koi entry signal nahi")

# State save
save_state(state)
print(f"   State saved!")

# Daily summary — 00:00, 08:00, 16:00 IST pe
hour_ist = datetime.now(IST).hour
if hour_ist in [0, 8, 16]:
    log_file = 'sp_trades_log.csv'
    if os.path.exists(log_file):
        log_df   = pd.read_csv(log_file)
        today    = datetime.now(IST).strftime("%Y-%m-%d")
        today_df = log_df[log_df['date'].str.startswith(today)]
        closed   = today_df[today_df['result'].isin(['WIN', 'LOSS'])]
        wins     = len(closed[closed['result'] == 'WIN'])
        losses   = len(closed[closed['result'] == 'LOSS'])
        total    = len(closed)
        wr       = f"{(wins/total)*100:.1f}%" if total > 0 else "N/A"
        send_telegram(
            f"📊 DAILY SUMMARY ({today})\n"
            f"📈 Trades  : {total}\n"
            f"✅ Wins    : {wins}\n"
            f"❌ Losses  : {losses}\n"
            f"🎯 Win Rate: {wr}\n"
            f"💼 Balance : ${state['balance']:,.2f}"
        )
        send_telegram_file(log_file, caption="📁 Trade Log CSV")

print(f"\n{'='*50}")
print(f"DONE! Balance: ${state['balance']:,.2f}")
print(f"{'='*50}\n")
