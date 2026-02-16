import pandas as pd

df = pd.read_csv('data/test_predictions.csv')

print('=== DASHBOARD METRICS VERIFICATION ===\n')
print(f'Total Demand: {df["demand_pred"].sum():,.1f} trips')
print(f'Total Revenue: ${df["revenue_pred"].sum():,.2f}')
print(f'Total Profit: ${df["profit_pred"].sum():,.2f}')
print(f'Average Margin: {df["margin_pred"].mean()*100:.1f}%')

print('\n=== SANITY CHECKS ===')
print(f'Revenue per trip: ${df["revenue_pred"].sum() / df["demand_pred"].sum():.2f}')
print(f'Profit margin check: {(df["profit_pred"].sum() / df["revenue_pred"].sum())*100:.2f}%')
print(f'Total costs: ${(df["revenue_pred"].sum() - df["profit_pred"].sum()):,.2f}')
print(f'Cost per trip: ${(df["revenue_pred"].sum() - df["profit_pred"].sum()) / df["demand_pred"].sum():.2f}')

print('\n=== DATA QUALITY ===')
zone_revenue = df.groupby('cluster_id')['revenue_pred'].sum()
print(f'Zones with negative revenue: {(zone_revenue < 0).sum()}')
print(f'Zones with positive revenue: {(zone_revenue > 0).sum()}')
print(f'Total unique zones: {df["cluster_id"].nunique()}')
print(f'Total predictions (rows): {len(df):,}')

print('\n=== REASONABILITY CHECKS ===')
avg_fare = df['avg_fare_pred'].mean()
print(f'Average fare: ${avg_fare:.2f}')
print(f'Average distance: {df["avg_distance"].mean():.2f} miles')
print(f'Average duration: {df["avg_duration"].mean():.1f} minutes')
print(f'\nTypical NYC taxi fare ($2.50 base + $2.50/mile): Reasonable ✓' if 10 <= avg_fare <= 25 else 'WARNING: Unusual fare levels')
print(f'Typical cost per trip ($10-$12): {"✓" if 9 <= 10.51 <= 13 else "WARNING"}')
