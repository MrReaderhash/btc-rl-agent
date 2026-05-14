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
TELEGRAM_TOKEN = "8644074664:AAH-pc-pp4FUpEKsyyUb5rEabiU7eDWfC2Q"   # BotFather se mila token
CHAT_ID        = "711544016"     # @userinfobot se mila ID

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print(f"   Telegram message error: {e}")

def send_telegram_file(filepath, caption=""):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        with open(filepath, 'rb') as f:
            requests.post(url, data={"chat_id": CHAT_ID, "caption": caption},
                          files={"document": f}, timeout=15)
    except Exception as e:
        print(f"   Telegram file error: {e}")

# =====================
# KNOWLEDGE BASE LOAD
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
    else:
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
# ENVIRONMENT
# =====================
class SPTradingEnv(gym.Env):
    def __init__(self, df, knowledge=None):
        super().__init__()
        self.df        = df.reset_index(drop=True)
        self.knowledge = knowledge
        self.current_step = 60

        log_file = 'sp_trades_log.csv'
        try:
            if os.path.exists(log_file):
                old_log = pd.read_csv(log_file)
                if len(old_log) > 0:
                    last_balance = float(old_log['balance'].iloc[-1])
                    self.balance = last_balance
                    self.initial_balance = last_balance
                    print(f"   Balance restored: ${last_balance:.2f}")
                else:
                    self.balance = 10000.0
                    self.initial_balance = 10000.0
            else:
                self.balance = 10000.0
                self.initial_balance = 10000.0
        except:
            self.balance = 10000.0
            self.initial_balance = 10000.0

        self.position            = 0
        self.entry_price         = 0.0
        self.stop_loss           = 0.0
        self.trailing_sl         = 0.0
        self.initial_sl_distance = 0.0
        self.position_size       = 0.0
        self.hold_count          = 0
        self.cooldown            = 0
        self.total_trades        = 0
        self.winning_trades      = 0
        self.max_trades_per_session = 5

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(185,), dtype=np.float32
        )

    def _get_obs(self):
        window       = self.df[['open', 'high', 'low', 'close', 'volume']].values
        candles_ohlc = self.df[['open', 'high', 'low', 'close']].values

        candles_20   = candles_ohlc[self.current_step-20:self.current_step]
        normalized   = (candles_20 - candles_20.mean(axis=0)) / (candles_20.std(axis=0) + 1e-8)
        ohlc_flat    = normalized.flatten()
        candle_feats = np.array(get_candle_features(candles_20), dtype=np.float32)

        current_price = candles_ohlc[self.current_step-1, 3]

        high_10 = candles_ohlc[self.current_step-10:self.current_step, 1].max()
        low_10  = candles_ohlc[self.current_step-10:self.current_step, 2].min()
        high_20 = candles_ohlc[self.current_step-20:self.current_step, 1].max()
        low_20  = candles_ohlc[self.current_step-20:self.current_step, 2].min()
        high_50 = candles_ohlc[self.current_step-50:self.current_step, 1].max()
        low_50  = candles_ohlc[self.current_step-50:self.current_step, 2].min()
        price_range = high_50 - low_50 + 1e-8

        breakout_feats = np.array([
            (current_price - high_10) / price_range,
            (current_price - low_10)  / price_range,
            (current_price - high_20) / price_range,
            (current_price - low_20)  / price_range,
            (current_price - high_50) / price_range,
            (current_price - low_50)  / price_range,
        ], dtype=np.float32)

        market_structure = get_market_structure(self.df, self.current_step)

        closes     = candles_ohlc[self.current_step-10:self.current_step, 3]
        mom_3      = (closes[-1] - closes[-3])  / (closes[-3]  + 1e-8)
        mom_5      = (closes[-1] - closes[-5])  / (closes[-5]  + 1e-8)
        mom_10     = (closes[-1] - closes[-10]) / (closes[-10] + 1e-8)
        volatility = closes.std() / (current_price + 1e-8)

        vol      = window[self.current_step-10:self.current_step, 4]
        vol_mean = vol.mean() + 1e-8
        win_rate = self.winning_trades / (self.total_trades + 1e-8)

        extra_feats = np.array([
            market_structure,
            mom_3, mom_5, mom_10,
            volatility,
            vol[-1] / vol_mean,
            vol[-3:].mean() / vol_mean,
            float(self.position),
            float(self.cooldown) / 5.0,
            float(self.hold_count) / 20.0,
            win_rate,
        ], dtype=np.float32)

        if self.position != 0 and self.initial_sl_distance > 0:
            sl_dist     = abs(current_price - self.trailing_sl) / (current_price + 1e-8)
            if self.position == 1:
                current_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                current_pnl = (self.entry_price - current_price) / self.entry_price
            current_r  = abs(current_price - self.entry_price) / (self.initial_sl_distance + 1e-8)
            locked_pnl = abs(self.trailing_sl - self.entry_price) / (self.initial_sl_distance + 1e-8)
        else:
            sl_dist = current_pnl = current_r = locked_pnl = 0.0

        trade_feats = np.array([sl_dist, locked_pnl, current_r, current_pnl], dtype=np.float32)

        obs = np.concatenate([
            ohlc_flat, candle_feats, breakout_feats, extra_feats, trade_feats
        ]).astype(np.float32)

        if len(obs) < 185:
            obs = np.pad(obs, (0, 185 - len(obs)))
        return obs[:185]

    def _calc_position_size(self, entry_price, sl_price):
        sl_distance_pct = abs(entry_price - sl_price) / entry_price
        if sl_distance_pct <= 0:
            return 0.0
        position_size = 0.005 / sl_distance_pct
        return min(position_size, 5.0)

    def _update_trailing_sl_long(self, current_price):
        r1 = self.initial_sl_distance
        if current_price >= self.entry_price + 4 * r1:
            new_trail = self.entry_price + 2 * r1
        elif current_price >= self.entry_price + 2 * r1:
            new_trail = self.entry_price
        else:
            new_trail = self.trailing_sl
        self.trailing_sl = max(self.trailing_sl, new_trail)

    def _update_trailing_sl_short(self, current_price):
        r1 = self.initial_sl_distance
        if current_price <= self.entry_price - 4 * r1:
            new_trail = self.entry_price - 2 * r1
        elif current_price <= self.entry_price - 2 * r1:
            new_trail = self.entry_price
        else:
            new_trail = self.trailing_sl
        self.trailing_sl = min(self.trailing_sl, new_trail)

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward        = 0
        forced_close  = False

        if self.cooldown > 0:
            self.cooldown -= 1
            action = 0

        at_daily_limit = (self.total_trades >= self.max_trades_per_session)
        market_str     = get_market_structure(self.df, self.current_step)

        if action == 1 and self.position == 0:
            if at_daily_limit:
                reward = -0.05
            else:
                swing_sl    = find_swing_low(self.df, self.current_step)
                sl_distance = current_price - swing_sl
                trend_bonus = 0.01 if market_str == 1.0 else -0.01
                if sl_distance <= 0:
                    reward = -0.01
                else:
                    pos_size                 = self._calc_position_size(current_price, swing_sl)
                    self.position            = 1
                    self.entry_price         = current_price
                    self.stop_loss           = swing_sl
                    self.trailing_sl         = swing_sl
                    self.initial_sl_distance = sl_distance
                    self.position_size       = pos_size
                    self.hold_count          = 0
                    self.total_trades       += 1
                    reward = trend_bonus

        elif action == 2 and self.position == 0:
            if at_daily_limit:
                reward = -0.05
            else:
                swing_sl    = find_swing_high(self.df, self.current_step)
                sl_distance = swing_sl - current_price
                trend_bonus = 0.01 if market_str == -1.0 else -0.01
                if sl_distance <= 0:
                    reward = -0.01
                else:
                    pos_size                 = self._calc_position_size(current_price, swing_sl)
                    self.position            = -1
                    self.entry_price         = current_price
                    self.stop_loss           = swing_sl
                    self.trailing_sl         = swing_sl
                    self.initial_sl_distance = sl_distance
                    self.position_size       = pos_size
                    self.hold_count          = 0
                    self.total_trades       += 1
                    reward = trend_bonus

        elif self.position != 0:
            self.hold_count += 1

            if self.position == 1:
                self._update_trailing_sl_long(current_price)
                if current_price <= self.trailing_sl:
                    profit_pct    = (current_price - self.entry_price) / self.entry_price
                    profit_amount = self.position_size * profit_pct
                    reward        = profit_amount * 150
                    self.balance *= (1 + profit_amount)
                    if profit_amount > 0:
                        self.winning_trades += 1
                    forced_close = True
                elif action == 0 and current_price > self.entry_price:
                    reward = 0.005
                elif action == 0 and current_price < self.entry_price:
                    reward = -0.005

            elif self.position == -1:
                self._update_trailing_sl_short(current_price)
                if current_price >= self.trailing_sl:
                    profit_pct    = (self.entry_price - current_price) / self.entry_price
                    profit_amount = self.position_size * profit_pct
                    reward        = profit_amount * 150
                    self.balance *= (1 + profit_amount)
                    if profit_amount > 0:
                        self.winning_trades += 1
                    forced_close = True
                elif action == 0 and current_price < self.entry_price:
                    reward = 0.005
                elif action == 0 and current_price > self.entry_price:
                    reward = -0.005

            if action == 3 and not forced_close:
                if self.position == 1:
                    profit_pct = (current_price - self.entry_price) / self.entry_price
                else:
                    profit_pct = (self.entry_price - current_price) / self.entry_price
                profit_amount = self.position_size * profit_pct
                reward        = profit_amount * (80 if profit_pct > 0 else 100)
                self.balance *= (1 + profit_amount)
                if profit_pct > 0:
                    self.winning_trades += 1
                forced_close = True

            if forced_close:
                self.position            = 0
                self.entry_price         = 0.0
                self.stop_loss           = 0.0
                self.trailing_sl         = 0.0
                self.initial_sl_distance = 0.0
                self.position_size       = 0.0
                self.hold_count          = 0
                self.cooldown            = 2

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None):
        self.current_step        = 60
        self.balance             = self.initial_balance
        self.position            = 0
        self.entry_price         = 0.0
        self.stop_loss           = 0.0
        self.trailing_sl         = 0.0
        self.initial_sl_distance = 0.0
        self.position_size       = 0.0
        self.hold_count          = 0
        self.cooldown            = 0
        self.total_trades        = 0
        self.winning_trades      = 0
        self.max_trades_per_session = 5
        return self._get_obs(), {}

# =====================
# MAIN
# =====================
now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
print(f"\n{'='*50}")
print(f"SUBHASHISH PANI RL AGENT v3")
print(f"Time: {now_ist}")
print(f"{'='*50}")

# Telegram: agent start hua
send_telegram(f"🤖 SP Agent v3 Started\n⏰ {now_ist}")

knowledge = load_knowledge()

print("\n1. BTC data fetch ho raha hai...")
try:
    exchange = ccxt.kraken()
    ohlcv    = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=500)
    print("   Kraken se data mila!")
except Exception as e:
    send_telegram(f"❌ Data fetch failed!\n{e}")
    print(f"   Error: {e}")
    raise

new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')

data_file = 'sp_btc_data.csv'
if os.path.exists(data_file):
    old_df      = pd.read_csv(data_file)
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates('timestamp').reset_index(drop=True)
    combined_df.to_csv(data_file, index=False)
    train_df = combined_df
else:
    new_df.to_csv(data_file, index=False)
    train_df = new_df

print(f"   Total candles: {len(train_df)}")

print("\n2. Training ho raha hai...")
train_env  = SPTradingEnv(train_df, knowledge)
model_file = 'sp_rl_model'

if os.path.exists(f'{model_file}.zip'):
    model = PPO.load(model_file, env=train_env)
    print("   Purana model load — aage seekhega!")
else:
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=2048, batch_size=64)
    print("   Naya model bana!")

model.learn(total_timesteps=50000)
model.save(model_file)
print("   Model saved!")

print("\n3. Live paper trading...")
try:
    live_ohlcv = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=100)
except Exception as e:
    send_telegram(f"❌ Live data fetch failed!\n{e}")
    print(f"   Live data error: {e}")
    raise

live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms')
live_df = live_df.reset_index(drop=True)

live_env = SPTradingEnv(live_df, knowledge)
obs, _   = live_env.reset()

trades_today  = []
prev_position = 0
prev_sl       = 0.0
prev_trail_sl = 0.0
prev_entry    = 0.0

for _ in range(len(live_df) - 61):
    action, _ = model.predict(obs)
    prev_position = live_env.position
    prev_sl       = live_env.stop_loss
    prev_trail_sl = live_env.trailing_sl
    prev_entry    = live_env.entry_price
    obs, reward, done, _, _ = live_env.step(action)
    current_price = live_df['close'].iloc[live_env.current_step-1]
    now_ist       = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

    # LONG entry
    if action == 1 and prev_position == 0 and live_env.position == 1:
        trades_today.append({
            'date': now_ist, 'action': 'LONG',
            'price': current_price,
            'sl': round(live_env.stop_loss, 1),
            'trail_sl': round(live_env.trailing_sl, 1),
            'pos_size': round(live_env.position_size, 3),
            'profit_pct': 0,
            'balance': round(live_env.balance, 2),
            'result': 'OPEN'
        })
        send_telegram(
            f"🟢 LONG ENTRY\n"
            f"⏰ {now_ist}\n"
            f"💰 Price : ${current_price:,.1f}\n"
            f"🛑 SL    : ${live_env.stop_loss:,.1f}\n"
            f"📦 Size  : {live_env.position_size:.3f}x\n"
            f"💼 Balance: ${live_env.balance:,.2f}"
        )

    # SHORT entry
    elif action == 2 and prev_position == 0 and live_env.position == -1:
        trades_today.append({
            'date': now_ist, 'action': 'SHORT',
            'price': current_price,
            'sl': round(live_env.stop_loss, 1),
            'trail_sl': round(live_env.trailing_sl, 1),
            'pos_size': round(live_env.position_size, 3),
            'profit_pct': 0,
            'balance': round(live_env.balance, 2),
            'result': 'OPEN'
        })
        send_telegram(
            f"🔴 SHORT ENTRY\n"
            f"⏰ {now_ist}\n"
            f"💰 Price : ${current_price:,.1f}\n"
            f"🛑 SL    : ${live_env.stop_loss:,.1f}\n"
            f"📦 Size  : {live_env.position_size:.3f}x\n"
            f"💼 Balance: ${live_env.balance:,.2f}"
        )

    # Trade close
    elif prev_position != 0 and live_env.position == 0:
        close_label = 'LONG CLOSE' if prev_position == 1 else 'SHORT CLOSE'
        result      = 'WIN' if reward > 0 else 'LOSS'
        trades_today.append({
            'date': now_ist, 'action': close_label,
            'price': current_price,
            'sl': round(prev_sl, 1),
            'trail_sl': round(prev_trail_sl, 1),
            'pos_size': 0,
            'profit_pct': round(reward, 4),
            'balance': round(live_env.balance, 2),
            'result': result
        })
        emoji = "🎉 WIN" if result == 'WIN' else "😞 LOSS"
        send_telegram(
            f"{'✅' if result == 'WIN' else '❌'} {close_label} — {emoji}\n"
            f"⏰ {now_ist}\n"
            f"💰 Price  : ${current_price:,.1f}\n"
            f"📈 P&L    : {reward:.4f}\n"
            f"💼 Balance: ${live_env.balance:,.2f}"
        )

    if done:
        break

# ---- CSV LOG ----
log_file    = 'sp_trades_log.csv'
file_exists = os.path.exists(log_file)
with open(log_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'date', 'action', 'price', 'sl', 'trail_sl',
        'pos_size', 'profit_pct', 'balance', 'result'
    ])
    if not file_exists:
        writer.writeheader()
    writer.writerows(trades_today)

wins   = len([t for t in trades_today if t['result'] == 'WIN'])
losses = len([t for t in trades_today if t['result'] == 'LOSS'])
total  = len(trades_today)
win_rate_str = f"{(wins/total)*100:.1f}%" if total > 0 else "N/A"

print(f"\n{'='*50}")
print(f"COMPLETE!")
print(f"Trades  : {total}")
print(f"Wins    : {wins}")
print(f"Losses  : {losses}")
if total > 0:
    print(f"Win Rate: {win_rate_str}")
print(f"Balance : ${live_env.balance:.2f}")
print(f"Log     : sp_trades_log.csv")
print(f"{'='*50}\n")

# ---- TELEGRAM: SESSION SUMMARY + CSV ----
send_telegram(
    f"📊 SESSION COMPLETE\n"
    f"⏰ {now_ist}\n"
    f"📈 Trades  : {total}\n"
    f"✅ Wins    : {wins}\n"
    f"❌ Losses  : {losses}\n"
    f"🎯 Win Rate: {win_rate_str}\n"
    f"💼 Balance : ${live_env.balance:,.2f}"
)

if os.path.exists(log_file):
    send_telegram_file(log_file, caption="📁 Trade Log CSV")
