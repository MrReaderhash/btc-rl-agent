import ccxt
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import time

# =====================
# STEP 1: FULL DATA FETCH (2018 se ab tak)
# =====================
def get_full_btc_data():
    exchange = ccxt.binance()
    all_data = []
    
    # 2018 se start
    since = exchange.parse8601('2018-01-01T00:00:00Z')
    
    print("2018 se ab tak ka data fetch ho raha hai...")
    print("Ye 2-3 minute lagega...")
    
    while True:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        print(f"  Fetched: {len(all_data)} candles...", end='\r')
        time.sleep(0.5)  # API rate limit
        
        # Agar latest data aa gaya toh stop
        if len(ohlcv) < 1000:
            break
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').reset_index(drop=True)
    
    # Save karo taaki baar baar fetch na karna pade
    df.to_csv('btc_data_full.csv', index=False)
    print(f"\nData saved! Total: {len(df)} candles (~{len(df)//24//365} years)")
    return df

# =====================
# CANDLE FEATURES
# =====================
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
            reward = -0.01

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
# DATA LOAD
# =====================
if os.path.exists('btc_data_full.csv'):
    print("Saved data mil gaya — load ho raha hai...")
    df = pd.read_csv('btc_data_full.csv')
    print(f"Loaded: {len(df)} candles")
else:
    df = get_full_btc_data()

# =====================
# TRAIN / TEST SPLIT
# =====================
split = int(len(df) * 0.8)
train_df = df[:split].reset_index(drop=True)
test_df = df[split:].reset_index(drop=True)

print(f"\nTrain: {len(train_df)} candles")
print(f"Test : {len(test_df)} candles (UNSEEN)")

# =====================
# TRAIN
# =====================
print("\nTraining shuru ho raha hai...")
print("Ye 1-2 ghante lagenge — laptop on rakhna!")
print("Progress neeche dikh raha hai...\n")

train_env = BTCTradingEnv(train_df)

# Agar pehle se model saved hai toh load karo
if os.path.exists('btc_rl_memory.zip'):
    print("Purana model mila — wahan se aage seekhega!")
    model = PPO.load("btc_rl_memory", env=train_env)
else:
    print("Naya model bana raha hai...")
    model = PPO("MlpPolicy", train_env, verbose=1, n_steps=2048, batch_size=64)

model.learn(total_timesteps=1000000)
model.save("btc_rl_memory")
print("\nModel saved — btc_rl_memory.zip")

# =====================
# TEST
# =====================
print("\nTesting on UNSEEN data...")
test_env = BTCTradingEnv(test_df)
obs, _ = test_env.reset()
balance_history = [test_env.initial_balance]
trades = []

for _ in range(len(test_df) - 21):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = test_env.step(action)
    balance_history.append(test_env.balance)
    if reward != 0:
        trades.append(reward)
    if done:
        break

wins = len([t for t in trades if t > 0])
losses = len([t for t in trades if t < 0])
total_trades = len(trades)

print(f"\n{'='*45}")
print(f"Initial Balance : $10,000")
print(f"Final Balance   : ${test_env.balance:.2f}")
print(f"Profit/Loss     : ${test_env.balance - 10000:.2f}")
print(f"Return          : {((test_env.balance - 10000) / 10000) * 100:.2f}%")
print(f"Total Trades    : {total_trades}")
if total_trades > 0:
    print(f"Win Rate        : {(wins/total_trades)*100:.1f}%")
print(f"{'='*45}")

# Chart
plt.figure(figsize=(12,5))
color = 'green' if test_env.balance > 10000 else 'red'
plt.plot(balance_history, color=color, linewidth=1.5)
plt.title('BTC RL Agent - Full Historical Training (UNSEEN Test)')
plt.xlabel('Steps')
plt.ylabel('Balance ($)')
plt.axhline(y=10000, color='gray', linestyle='--', label='Starting Balance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('performance_full.png')
plt.show()
print("\nDone! Ab update.py banayenge kal.")