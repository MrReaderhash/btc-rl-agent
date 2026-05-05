import ccxt
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os
import csv
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

# =====================
# SWING HIGH/LOW FINDER
# =====================
def find_swing_low(df, current_step, lookback=10):
    lows = df['low'].iloc[current_step-lookback:current_step]
    return lows.min()

def find_swing_high(df, current_step, lookback=10):
    highs = df['high'].iloc[current_step-lookback:current_step]
    return highs.max()

def get_candle_features(candles):
    features = []
    for c in candles:
        o, h, l, cl = c[0], c[1], c[2], c[3]
        body = cl - o
        body_size = abs(body) / (h - l + 1e-8)
        upper_wick = (h - max(o, cl)) / (h - l + 1e-8)
        lower_wick = (min(o, cl) - l) / (h - l + 1e-8)
        direction = 1.0 if body > 0 else -1.0
        features.extend([body_size, upper_wick, lower_wick, direction])
    return features

# =====================
# ENVIRONMENT
# =====================
class BTCTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
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

        # Actions: 0=Hold, 1=Long, 2=Short, 3=Close
        self.action_space = spaces.Discrete(4)

        # 180 observations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(180,), dtype=np.float32
        )

    def _calculate_sl_target(self, action, current_price):
        """SL aur Target calculate karo"""
        if action == 1:  # Long
            sl = find_swing_low(self.df, self.current_step)
            sl_distance = current_price - sl
            target = current_price + (sl_distance * 3)  # 1:3 RR
        else:  # Short
            sl = find_swing_high(self.df, self.current_step)
            sl_distance = sl - current_price
            target = current_price - (sl_distance * 3)  # 1:3 RR
        return sl, target

    def _get_obs(self):
        window = self.df[['open', 'high', 'low', 'close', 'volume']].values
        candles_ohlc = self.df[['open', 'high', 'low', 'close']].values

        # Last 20 candles OHLC normalized
        candles_20 = candles_ohlc[self.current_step-20:self.current_step]
        normalized = (candles_20 - candles_20.mean(axis=0)) / (candles_20.std(axis=0) + 1e-8)
        ohlc_flat = normalized.flatten()  # 80

        # Candle features
        candle_feats = np.array(get_candle_features(candles_20), dtype=np.float32)  # 80

        current_price = candles_ohlc[self.current_step-1, 3]

        # Breakout levels
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
        ], dtype=np.float32)  # 6

        # Momentum
        closes = candles_ohlc[self.current_step-10:self.current_step, 3]
        mom_3  = (closes[-1] - closes[-3])  / (closes[-3]  + 1e-8)
        mom_5  = (closes[-1] - closes[-5])  / (closes[-5]  + 1e-8)
        mom_10 = (closes[-1] - closes[-10]) / (closes[-10] + 1e-8)
        momentum_feats = np.array([mom_3, mom_5, mom_10], dtype=np.float32)  # 3

        # Volatility
        volatility = closes.std() / (current_price + 1e-8)
        volatility_feat = np.array([volatility], dtype=np.float32)  # 1

        # Volume
        vol = window[self.current_step-10:self.current_step, 4]
        vol_mean = vol.mean() + 1e-8
        vol_feats = np.array([
            vol[-1] / vol_mean,
            vol[-3:].mean() / vol_mean,
            vol[-5:].mean() / vol_mean,
        ], dtype=np.float32)  # 3

        # SL/Target/RR features
        if self.position != 0 and self.stop_loss > 0:
            sl_distance  = abs(current_price - self.stop_loss)  / (current_price + 1e-8)
            tgt_distance = abs(self.target - current_price)     / (current_price + 1e-8)
            rr_ratio     = tgt_distance / (sl_distance + 1e-8)
            trail_dist   = abs(current_price - self.trailing_sl) / (current_price + 1e-8)

            # Current profit/loss %
            if self.position == 1:
                current_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                current_pnl = (self.entry_price - current_price) / self.entry_price
        else:
            sl_distance  = 0.0
            tgt_distance = 0.0
            rr_ratio     = 0.0
            trail_dist   = 0.0
            current_pnl  = 0.0

        trade_feats = np.array([
            float(self.position),
            sl_distance,
            tgt_distance,
            rr_ratio,
            trail_dist,
            current_pnl,
            float(self.hold_count) / 20.0,
            float(self.cooldown)   / 5.0,
        ], dtype=np.float32)  # 8

        # Total = 80+80+6+3+1+3+8 = 181 → slice to 180
        obs = np.concatenate([
            ohlc_flat,
            candle_feats,
            breakout_feats,
            momentum_feats,
            volatility_feat,
            vol_feats,
            trade_feats
        ]).astype(np.float32)

        return obs[:180]

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        forced_close = False

        # Cooldown — force hold
        if self.cooldown > 0:
            self.cooldown -= 1
            action = 0

        # Overtrading penalty
        self.recent_trades.append(1 if action in [1, 2] else 0)
        if len(self.recent_trades) > 5:
            self.recent_trades.pop(0)
        overtrade_penalty = -0.05 if sum(self.recent_trades) >= 3 else 0

        # =====================
        # LONG ENTRY
        # =====================
        if action == 1 and self.position == 0:
            sl, target = self._calculate_sl_target(1, current_price)
            sl_distance = current_price - sl

            # 1:3 RR check — agar nahi banta toh trade mat lo
            if sl_distance <= 0 or sl_distance > current_price * 0.05:
                reward = -0.004  # Invalid setup
            else:
                self.position = 1
                self.entry_price = current_price
                self.stop_loss = sl
                self.target = target
                self.trailing_sl = sl
                self.hold_count = 0
                reward = overtrade_penalty
                print(f"   LONG @ {current_price:.0f} | SL: {sl:.0f} | Target: {target:.0f}")

        # =====================
        # SHORT ENTRY
        # =====================
        elif action == 2 and self.position == 0:
            sl, target = self._calculate_sl_target(2, current_price)
            sl_distance = sl - current_price

            if sl_distance <= 0 or sl_distance > current_price * 0.05:
                reward = -0.004
            else:
                self.position = -1
                self.entry_price = current_price
                self.stop_loss = sl
                self.target = target
                self.trailing_sl = sl
                self.hold_count = 0
                reward = overtrade_penalty
                print(f"   SHORT @ {current_price:.0f} | SL: {sl:.0f} | Target: {target:.0f}")

        # =====================
        # POSITION MANAGEMENT
        # =====================
        elif self.position != 0:
            self.hold_count += 1

            if self.position == 1:
                # Trailing SL update — profit badhne pe SL upar
                if current_price > self.entry_price:
                    new_trail = current_price - (self.entry_price - self.stop_loss)
                    self.trailing_sl = max(self.trailing_sl, new_trail)

                # SL hit — loss jaldi cut karo
                if current_price <= self.trailing_sl:
                    profit = (current_price - self.entry_price) / self.entry_price
                    reward = profit * 150  # Extra punishment for loss
                    self.balance *= (1 + profit)
                    forced_close = True
                    print(f"   SL HIT LONG @ {current_price:.0f} | P&L: {profit*100:.2f}%")

                # Target hit — profit book karo
                elif current_price >= self.target:
                    profit = (current_price - self.entry_price) / self.entry_price
                    reward = profit * 150  # Bonus for hitting target
                    self.balance *= (1 + profit)
                    forced_close = True
                    print(f"   TARGET HIT LONG @ {current_price:.0f} | P&L: {profit*100:.2f}%")

            elif self.position == -1:
                # Trailing SL update for short
                if current_price < self.entry_price:
                    new_trail = current_price + (self.stop_loss - self.entry_price)
                    self.trailing_sl = min(self.trailing_sl, new_trail)

                # SL hit
                if current_price >= self.trailing_sl:
                    profit = (self.entry_price - current_price) / self.entry_price
                    reward = profit * 150
                    self.balance *= (1 + profit)
                    forced_close = True
                    print(f"   SL HIT SHORT @ {current_price:.0f} | P&L: {profit*100:.2f}%")

                # Target hit
                elif current_price <= self.target:
                    profit = (self.entry_price - current_price) / self.entry_price
                    reward = profit * 150
                    self.balance *= (1 + profit)
                    forced_close = True
                    print(f"   TARGET HIT SHORT @ {current_price:.0f} | P&L: {profit*100:.2f}%")

            # Manual close by agent
            if action == 3 and not forced_close:
                if self.position == 1:
                    profit = (current_price - self.entry_price) / self.entry_price
                else:
                    profit = (self.entry_price - current_price) / self.entry_price

                # Profit run kar raha tha — thoda penalty
                if profit > 0.01 and current_price < self.target:
                    reward = profit * 80  # Less reward — target nahi aaya
                elif profit < 0 and current_price > self.stop_loss:
                    reward = profit * 100  # Normal loss
                else:
                    reward = profit * 100

                self.balance *= (1 + profit)
                forced_close = True

            # Holding reward/penalty
            if not forced_close and action == 0:
                if self.position == 1 and current_price > self.entry_price:
                    reward = 0.005  # Profit mein hold — good!
                elif self.position == -1 and current_price < self.entry_price:
                    reward = 0.005
                else:
                    reward = -0.004  # Loss mein hold — bad!

            if forced_close:
                self.position = 0
                self.entry_price = 0.0
                self.stop_loss = 0.0
                self.target = 0.0
                self.trailing_sl = 0.0
                self.hold_count = 0
                self.cooldown = 3

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None):
        self.current_step = 60
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.target = 0.0
        self.trailing_sl = 0.0
        self.hold_count = 0
        self.cooldown = 0
        self.recent_trades = []
        return self._get_obs(), {}

# =====================
# STEP 1: NAYA DATA
# =====================
now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
print(f"\n{'='*50}")
print(f"UPDATE: {now_ist}")
print(f"{'='*50}")

print("\n1. Naya BTC data fetch ho raha hai...")
try:
    exchange = ccxt.kraken()
    ohlcv = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=500)
    print("   Kraken se data mila!")
except:
    print("   Kraken fail — Binance try kar raha hai...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=500)
    print("   Binance se data mila!")
new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')

if os.path.exists('btc_data_full.csv'):
    old_df = pd.read_csv('btc_data_full.csv')
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates('timestamp').reset_index(drop=True)
    combined_df.to_csv('btc_data_full.csv', index=False)
    train_df = combined_df
    print(f"   Total candles: {len(combined_df)}")
else:
    new_df.to_csv('btc_data_full.csv', index=False)
    train_df = new_df

# =====================
# STEP 2: MODEL UPDATE
# =====================
print("\n2. Model seekh raha hai...")
train_env = BTCTradingEnv(train_df)

print("   Naya improved model bana raha hai...")
model = PPO("MlpPolicy", train_env, verbose=0, n_steps=2048, batch_size=64)
model.learn(total_timesteps=330000)
model.save("btc_rl_memory")
print("   Model saved!")

# =====================
# STEP 3: LIVE PAPER TRADING
# =====================
print("\n3. Live market data fetch kar raha hai...")
try:
    live_ohlcv = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=100)
except:
    exchange = ccxt.binance()
    live_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms')
live_df = live_df.reset_index(drop=True)
print(f"   Live candles: {len(live_df)}")

live_env = BTCTradingEnv(live_df)
obs, _ = live_env.reset()

trades_today = []
prev_position = 0
prev_sl = 0.0
prev_target = 0.0

for _ in range(len(live_df) - 61):
    action, _ = model.predict(obs)
    prev_position = live_env.position
    prev_sl = live_env.stop_loss
    prev_target = live_env.target
    obs, reward, done, _, _ = live_env.step(action)
    current_price = live_df['close'].iloc[live_env.current_step-1]
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M")

    # Long entry
    if action == 1 and prev_position == 0 and live_env.position == 1:
        trades_today.append({
            'date': now_ist,
            'action': 'LONG',
            'price': current_price,
            'sl': round(live_env.stop_loss, 1),
            'target': round(live_env.target, 1),
            'profit_pct': 0,
            'balance': round(live_env.balance, 2),
            'result': 'OPEN'
        })

    # Short entry
    elif action == 2 and prev_position == 0 and live_env.position == -1:
        trades_today.append({
            'date': now_ist,
            'action': 'SHORT',
            'price': current_price,
            'sl': round(live_env.stop_loss, 1),
            'target': round(live_env.target, 1),
            'profit_pct': 0,
            'balance': round(live_env.balance, 2),
            'result': 'OPEN'
        })

    # Close
    elif prev_position != 0 and live_env.position == 0:
        close_label = 'LONG CLOSE' if prev_position == 1 else 'SHORT CLOSE'
        trades_today.append({
            'date': now_ist,
            'action': close_label,
            'price': current_price,
            'sl': round(prev_sl, 1),
            'target': round(prev_target, 1),
            'profit_pct': round(reward, 4),
            'balance': round(live_env.balance, 2),
            'result': 'WIN' if reward > 0 else 'LOSS'
        })

    if done:
        break

# =====================
# STEP 4: LOG
# =====================
print("\n4. Log save ho raha hai...")
log_file = 'trades_log.csv'
file_exists = os.path.exists(log_file)

with open(log_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['date', 'action', 'price', 'sl', 'target', 'profit_pct', 'balance', 'result'])
    if not file_exists:
        writer.writeheader()
    writer.writerows(trades_today)

wins   = len([t for t in trades_today if t['result'] == 'WIN'])
losses = len([t for t in trades_today if t['result'] == 'LOSS'])
total  = len(trades_today)

print(f"\n{'='*50}")
print(f"COMPLETE!")
print(f"Trades  : {total}")
print(f"Wins    : {wins}")
print(f"Losses  : {losses}")
if total > 0:
    print(f"Win Rate: {(wins/total)*100:.1f}%")
print(f"Balance : ${live_env.balance:.2f}")
print(f"Log     : trades_log.csv")
print(f"{'='*50}\n")
