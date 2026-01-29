import pandas as pd

df = pd.read_csv('outputs/scores.csv')

print('=== z_window configuration ===')
print('Nifty 50:  252 days (full)')
print('Midcap:    160 days (reduced)')
print('Smallcap:  160 days (reduced, same as midcap)')
print()

print('=== Latest scores (2026-01-29) ===')
latest = df[df['date'] == '2026-01-29'].sort_values('index')
for idx in ['nifty50', 'midcap100', 'smallcap100']:
    row = latest[latest['index'] == idx]
    if len(row) > 0:
        r = row.iloc[0]
        print(f'{idx:12s}: trend_score={r["trend_score"]:6.2f}, score={r["score"]:6.2f}, label={r["label"]}')
