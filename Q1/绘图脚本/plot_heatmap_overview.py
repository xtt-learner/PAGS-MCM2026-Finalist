import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- Style Settings (English for MCM) ---
sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_controversy_heatmap(csv_path):
    # 1. Path Check
    if not os.path.exists(csv_path):
        try:
            # Try auto-location based on standard project structure
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Assuming script is in root or a subfolder, try to find project root
            # This logic tries to go up until it finds 'data' directory
            project_root = current_dir
            while not os.path.exists(os.path.join(project_root, "data")) and os.path.dirname(
                    project_root) != project_root:
                project_root = os.path.dirname(project_root)

            auto_path = os.path.join(project_root, "data", "Q1data", "week_diagnostics.csv")
            if os.path.exists(auto_path):
                csv_path = auto_path
            else:
                print(f"❌ File not found at {csv_path} or auto-location failed.")
                return
        except:
            print(f"❌ Path error.")
            return

    print(f"Reading data: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # 2. Pivot the Data
    heatmap_data = df.pivot(index="season", columns="week", values="accept_rate")

    # 3. Create the Plot
    # Keep the large size to make numbers readable
    fig = plt.figure(figsize=(18, 18))

    # Color Map: Red (Low) -> Blue (High)
    cmap = sns.color_palette("RdYlBu", as_cmap=True)

    # 4. Draw Heatmap with Annotations
    ax = sns.heatmap(heatmap_data,
                     cmap=cmap,
                     annot=True,  # Show numbers
                     fmt=".2f",  # Format: 0.12
                     annot_kws={"size": 8},  # Small font for numbers
                     linewidths=.5,
                     linecolor='whitesmoke',
                     # Use a separate axes for colorbar to better control layout if needed,
                     # but standard cbar_kws usually works okay. Let's shrink it a bit more.
                     cbar_kws={'label': 'Model Acceptance Rate (Red = High Controversy)', 'shrink': 0.4, 'aspect': 30},
                     vmin=0.0, vmax=0.5)

    # 5. Formatting Labels
    # Increase padding (pad=20) to push titles away from the plot
    plt.title("Full History of Model Acceptance Rates (Values inside cells)", fontsize=18, fontweight='bold', pad=30)
    plt.xlabel("Competition Week", fontsize=14, labelpad=15)
    plt.ylabel("Season", fontsize=14, labelpad=15)

    plt.yticks(rotation=0)

    # =================================================================
    # 🛠️ 关键修复：手动调整边距，防止遮挡
    # =================================================================
    # top=0.93: 意味着图像顶部只占用到画布 93% 的位置，留出 7% 给大标题
    # bottom=0.1: 底部留出 10% 给 X 轴标签
    # left=0.08: 左侧留出 8% 给 Y 轴标签
    plt.subplots_adjust(top=0.93, bottom=0.1, left=0.08, right=0.98)

    # 去掉 tight_layout，因为我们已经手动调整了
    # plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Standard path attempt (assuming script in root)
    csv_path = os.path.join("data", "Q1data", "week_diagnostics.csv")
    plot_controversy_heatmap(csv_path)