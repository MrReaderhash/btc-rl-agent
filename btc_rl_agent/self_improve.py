import pandas as pd
import json
import os
import re
from datetime import datetime
from groq import Groq

GROQ_API_KEY = "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"

# =====================
# STEP 1: PERFORMANCE ANALYZE KARO
# =====================
def analyze_performance():
    if not os.path.exists('trades_log.csv'):
        return None
    
    df = pd.read_csv('trades_log.csv', on_bad_lines='skip')
    
    # Last 50 trades analyze karo
    recent = df.tail(50)
    
    total = len(recent)
    wins = len(recent[recent['result'] == 'WIN'])
    losses = len(recent[recent['result'] == 'LOSS'])
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # Balance trend
    if len(df) > 10:
        recent_balance = df['balance'].tail(10).values
        balance_trend = "improving" if recent_balance[-1] > recent_balance[0] else "declining"
    else:
        balance_trend = "unknown"
    
    # Current balance
    current_balance = df['balance'].iloc[-1] if len(df) > 0 else 10000
    
    performance = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "balance_trend": balance_trend,
        "current_balance": current_balance,
        "initial_balance": 10000,
        "pnl": round(current_balance - 10000, 2)
    }
    
    print(f"\nPerformance Analysis:")
    print(f"Win Rate    : {win_rate:.1f}%")
    print(f"Balance     : ${current_balance:.2f}")
    print(f"P&L         : ${performance['pnl']:.2f}")
    print(f"Trend       : {balance_trend}")
    
    return performance

# =====================
# STEP 2: CURRENT PARAMS PADHNA
# =====================
def get_current_params():
    if not os.path.exists('update.py'):
        return None
    
    with open('update.py', 'r') as f:
        content = f.read()
    
    # Current values nikalo
    params = {}
    
    # Training steps
    match = re.search(r'total_timesteps=(\d+)', content)
    params['timesteps'] = int(match.group(1)) if match else 10000
    
    # Penalty
    match = re.search(r'reward = (-[\d.]+)', content)
    params['penalty'] = float(match.group(1)) if match else -0.01
    
    print(f"\nCurrent Params:")
    print(f"Timesteps : {params['timesteps']}")
    print(f"Penalty   : {params['penalty']}")
    
    return params, content

# =====================
# STEP 3: AI SE DECISION LO
# =====================
def get_ai_decision(performance, params):
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
You are an AI trading agent optimizer. Analyze this performance and suggest parameter changes.

Current Performance:
- Win Rate: {performance['win_rate']}%
- Balance Trend: {performance['balance_trend']}
- Current Balance: ${performance['current_balance']}
- P&L: ${performance['pnl']}

Current Parameters:
- Training Timesteps: {params['timesteps']}
- Hold Penalty: {params['penalty']}

Rules:
- If win_rate < 30%: increase timesteps by 10000, reduce penalty by 0.001
- If win_rate > 50%: keep timesteps same, slight increase penalty
- If balance declining: increase timesteps, reduce penalty
- If balance improving: keep same or slight adjustments
- Timesteps should be between 10000 and 100000
- Penalty should be between -0.0001 and -0.01

Respond ONLY in this JSON format:
{{
    "new_timesteps": <number>,
    "new_penalty": <number>,
    "reason": "<one line explanation>"
}}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    
    text = response.choices[0].message.content
    start = text.find('{')
    end = text.rfind('}') + 1
    decision = json.loads(text[start:end])
    
    print(f"\nAI Decision:")
    print(f"New Timesteps : {decision['new_timesteps']}")
    print(f"New Penalty   : {decision['new_penalty']}")
    print(f"Reason        : {decision['reason']}")
    
    return decision

# =====================
# STEP 4: UPDATE.PY KHUD UPDATE KARO
# =====================
def update_params(content, decision):
    # Timesteps update
    new_content = re.sub(
        r'total_timesteps=\d+',
        f"total_timesteps={decision['new_timesteps']}",
        content
    )
    
    # Penalty update
    new_content = re.sub(
        r'reward = -[\d.]+',
        f"reward = {decision['new_penalty']}",
        new_content
    )
    
    with open('update.py', 'w') as f:
        f.write(new_content)
    
    print("\nupdate.py successfully updated!")
    return new_content

# =====================
# STEP 5: LOG KARO
# =====================
def save_improvement_log(performance, params, decision):
    log_entry = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "win_rate": performance['win_rate'],
        "balance": performance['current_balance'],
        "pnl": performance['pnl'],
        "trend": performance['balance_trend'],
        "old_timesteps": params['timesteps'],
        "new_timesteps": decision['new_timesteps'],
        "old_penalty": params['penalty'],
        "new_penalty": decision['new_penalty'],
        "reason": decision['reason']
    }
    
    log_file = 'improvement_log.json'
    logs = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"Improvement log saved!")

# =====================
# MAIN
# =====================
print("="*50)
print(f"SELF IMPROVE: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*50)

# Performance check
performance = analyze_performance()
if not performance:
    print("No trades log found — skipping!")
    exit()

# Current params
result = get_current_params()
if not result:
    print("update.py not found!")
    exit()

params, content = result

# AI decision
decision = get_ai_decision(performance, params)

# Update karo
update_params(content, decision)

# Log karo
save_improvement_log(performance, params, decision)

print("\n" + "="*50)
print("SELF IMPROVEMENT COMPLETE!")
print("Agent ne apne aap ko update kar liya!")
print("="*50)
