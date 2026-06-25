import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

def run_eda(data_path, output_dir):
    print(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("Generating Traffic Demand Distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['traffic_demand'], kde=True, bins=50, color='#1f77b4')
    plt.title("Traffic Demand Distribution (Vehicles/Hour)", pad=15)
    plt.xlabel("Traffic Demand")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "traffic_demand_distribution.png"), dpi=150)
    plt.close()
    
    print("Generating Hourly Traffic Analysis...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='hour', y='traffic_demand', errorbar=('ci', 95), color='#2ca02c', linewidth=2)
    plt.title("Hourly Traffic Demand Profile (with 95% Confidence Interval)", pad=15)
    plt.xlabel("Hour of Day")
    plt.ylabel("Mean Traffic Demand")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hourly_traffic_analysis.png"), dpi=150)
    plt.close()
    
    print("Generating Weather Impact Analysis...")
    plt.figure(figsize=(10, 6))
    weather_order = df.groupby('weather_condition')['traffic_demand'].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x='weather_condition', y='traffic_demand', errorbar='ci', order=weather_order, hue='weather_condition', legend=False, palette='viridis')
    plt.title("Mean Traffic Demand by Weather Condition", pad=15)
    plt.xlabel("Weather Condition")
    plt.ylabel("Mean Traffic Demand")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weather_impact_analysis.png"), dpi=150)
    plt.close()
    
    print("Generating Correlation Heatmap...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Numerical Feature Correlation Heatmap", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    
    print("Generating Road Type Distribution...")
    plt.figure(figsize=(10, 6))
    road_order = df['road_type'].value_counts().index
    sns.countplot(data=df, x='road_type', order=road_order, hue='road_type', legend=False, palette='muted')
    plt.title("Road Type Distribution", pad=15)
    plt.xlabel("Road Type")
    plt.ylabel("Record Count")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "road_type_distribution.png"), dpi=150)
    plt.close()
    
    print("Generating Box Plot - Demand by Road Type...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='road_type', y='traffic_demand', order=road_order, hue='road_type', legend=False, palette='Set2')
    plt.title("Traffic Demand Variability by Road Type", pad=15)
    plt.xlabel("Road Type")
    plt.ylabel("Traffic Demand (Vehicles/Hour)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "demand_by_road_type_boxplot.png"), dpi=150)
    plt.close()
    
    print("Generating Large Vehicles vs. Demand Scatter Plot...")
    plt.figure(figsize=(10, 6))
    # Using hexbin or sample scatter for dense plot to avoid lag
    if len(df) > 10000:
        sample_df = df.sample(10000, random_state=42)
    else:
        sample_df = df
    sns.scatterplot(data=sample_df, x='large_vehicles_count', y='traffic_demand', alpha=0.3, color='#e377c2')
    plt.title("Large Vehicles Count vs. Traffic Demand (Sampled 10k Rows)", pad=15)
    plt.xlabel("Large Vehicles Count")
    plt.ylabel("Traffic Demand")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "large_vehicles_vs_demand.png"), dpi=150)
    plt.close()
    
    print("Generating Monthly / Day-of-Week Heatmap...")
    pivot_df = df.groupby(['day_of_week', 'month'])['traffic_demand'].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap="YlGnBu", cbar=True)
    plt.title("Mean Traffic Demand by Month and Day of Week", pad=15)
    plt.xlabel("Month")
    plt.ylabel("Day of Week (0=Monday)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_dayofweek_heatmap.png"), dpi=150)
    plt.close()
    
    print("EDA completed successfully! Figures saved to", output_dir)

if __name__ == "__main__":
    run_eda("data/raw/smart_city_traffic_data.csv", "reports/figures")
