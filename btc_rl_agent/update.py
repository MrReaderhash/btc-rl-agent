import ccxt
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import os
import csv
from datetime import datetime

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

class BTCTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 20
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.position = 0
        self.entry_price = 0.0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(81,), dtype=np.float32
        )

    def _get_obs(self):
        window = self.df[['open', 'high', 'low', 'close']].values
        candles = window[self.current_step-10:self.current_step]
        normalized = (candles - candles.mean(axis=0)) / (candles.std(axis=0) + 1e-8)
        ohlc_flat = normalized.flatten()
        candle_feats = np.array(get_candle_features(candles), dtype=np.float32)
        position_feat = np.array([float(self.position)], dtype=np.float32)
        return np.concatenate([ohlc_flat, candle_feats, position_feat]).astype(np.float32)

    def step(self, action):
        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = current_price
        elif action == 3 and self.position != 0:
            if self.position == 1:
                profit = (current_price - self.entry_price) / self.entry_price
            else:
                profit = (self.entry_price - current_price) / self.entry_price
            reward = profit * 100
            self.balance *= (1 + profit)
            self.position = 0
            self.entry_price = 0.0
        elif action == 0 and self.position != 0:
            reward = -0.002
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None):
        self.current_step = 20
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs(), {}

# =====================
# STEP 1: NAYA DATA
# =====================
print(f"\n{'='*50}")
print(f"UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*50}")

print("\n1. Naya BTC data fetch ho raha hai...")
exchange = ccxt.kraken()
ohlcv = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=500)
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

if os.path.exists('btc_rl_memory.zip'):
    model = PPO.load("btc_rl_memory", env=train_env)
    print("   Purana model load — aage se seekhega!")
else:
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=2048)
    print("   Naya model bana!")

model.learn(total_timesteps=200000)
model.save("btc_rl_memory")
print("   Model saved!")

# =====================
# STEP 3: LIVE PAPER TRADING
# =====================
print("\n3. Live market data fetch kar raha hai...")
live_ohlcv = exchange.fetch_ohlcv('BTC/USD', timeframe='1h', limit=100)
live_df = pd.DataFrame(live_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], unit='ms')
live_df = live_df.reset_index(drop=True)
print(f"   Live candles: {len(live_df)}")

live_env = BTCTradingEnv(live_df)
obs, _ = live_env.reset()

trades_today = []
actions_map = {0: 'HOLD', 1: 'BUY', 2: 'SHORT', 3: 'CLOSE'}

for _ in range(len(live_df) - 21):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = live_env.step(action)
    if reward != 0:
        trades_today.append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'action': actions_map[int(action)],
            'price': live_df['close'].iloc[live_env.current_step-1],
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
    writer = csv.DictWriter(f, fieldnames=['date', 'action', 'price', 'profit_pct', 'balance', 'result'])
    if not file_exists:
        writer.writeheader()
    writer.writerows(trades_today)

wins = len([t for t in trades_today if t['result'] == 'WIN'])
losses = len([t for t in trades_today if t['result'] == 'LOSS'])
total = len(trades_today)

print(f"\n{'='*50}")
print(f"COMPLETE!")
print(f"Trades    : {total}")
print(f"Wins      : {wins}")
print(f"Losses    : {losses}")
if total > 0:
    print(f"Win Rate  : {(wins/total)*100:.1f}%")
print(f"Balance   : ${live_env.balance:.2f}")
print(f"Log saved : trades_log.csv")
print(f"{'='*50}\n")
