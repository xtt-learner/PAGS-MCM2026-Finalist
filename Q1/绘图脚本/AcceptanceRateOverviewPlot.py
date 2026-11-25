import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Settings ---
sns.set_theme(style="whitegrid")
# Use standard English fonts (Arial/Helvetica) instead of Chinese fonts
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_acceptance_rate_overview(csv_path, threshold=0.1):
    # 1. Smart Path Check
    if not os.path.exists(csv_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Try to locate the file again
        if os.path.exists(os.path.join("data", "Q1data", "week_diagnostics.csv")):
            csv_path = os.path.join("data", "Q1data", "week_diagnostics.csv")
        else:
            print(f"❌ File not found: {csv_path}")
            return

    print(f"Reading data: {csv_path} ...")
    df = pd.read_csv(csv_path)

    # 2. Data Preprocessing
    # Sort by season and week to create a continuous timeline
    df = df.sort_values(by=['season', 'week']).reset_index(drop=True)
    df['timeline_index'] = df.index

    # 3. Filter "High Controversy" weeks (Low Acceptance Rate)
    low_accept = df[df['accept_rate'] <= threshold].copy()

    # Print Statistics to Console
    print(f"🔍 Statistical Summary:")
    print(f"Total Weeks Processed: {len(df)}")
    print(f"Total 'High Controversy Weeks' (Rate < {threshold * 100}%): {len(low_accept)}")

    # 4. Plotting
    plt.figure(figsize=(15, 7))

    # A. Plot Background Trend Line (Gray)
    plt.plot(df['timeline_index'], df['accept_rate'], color='gray', alpha=0.3, linewidth=1,
             label='Acceptance Rate Trend')

    # B. Plot Normal Points (Blue)
    normal_points = df[df['accept_rate'] > threshold]
    plt.scatter(normal_points['timeline_index'], normal_points['accept_rate'],
                s=10, color='steelblue', alpha=0.6, label='Normal Weeks')

    # C. Plot Controversial Points (Red, highlighted)
    if not low_accept.empty:
        plt.scatter(low_accept['timeline_index'], low_accept['accept_rate'],
                    s=50, color='crimson', edgecolors='white', zorder=5, label=f'Controversial (Rate < {threshold})')

        # D. Annotate Controversial Points (e.g., S15-W3)
        for _, row in low_accept.iterrows():
            plt.text(row['timeline_index'], row['accept_rate'] + 0.02,
                     f"S{int(row['season'])}-W{int(row['week'])}",
                     fontsize=8, color='darkred', ha='center', fontweight='bold')

    # 5. Formatting Axes and Labels (English)
    season_starts = df.drop_duplicates(subset=['season'], keep='first')
    xticks_data = season_starts.iloc[::3]

    plt.xticks(xticks_data['timeline_index'], [f"S{int(s)}" for s in xticks_data['season']], rotation=0)

    # --- Key English Labels Here ---
    plt.xlabel("Season Timeline (Chronological)", fontsize=12)
    plt.ylabel("Model Acceptance Rate", fontsize=12)
    plt.title(f"Full-Season Controversy Monitoring: Moments with Acceptance Rate Below {int(threshold * 100)}%",
              fontsize=16)
    plt.legend(loc='upper right')

    # Add Threshold Line
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    project_root = os.path.dirname(os.path.dirname(current_script_dir))

    data_path = os.path.join(project_root, "data", "Q1data", "week_diagnostics.csv")

    plot_acceptance_rate_overview(data_path, threshold=0.08)