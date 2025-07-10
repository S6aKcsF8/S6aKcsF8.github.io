import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import warnings
import os
import glob

warnings.filterwarnings('ignore')

def create_final_chart():
    # CSVファイルを探す（タイムスタンプ付きファイルも考慮）
    csv_files = glob.glob('age_trend_data_for_looker*.csv')
    if not csv_files:
        print("エラー: age_trend_data_for_looker.csvファイルが見つかりません")
        return
    
    # 最新のファイルを使用
    latest_file = max(csv_files, key=os.path.getctime)
    print(f"使用するデータファイル: {latest_file}")
    
    # データを読み込み
    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # 実績と予測データを結合
    all_data = df.copy()
    
    # ピボットテーブルで横持ちに変換
    pivot_data = all_data.pivot(index='date', columns='age_label', values='value')
    
    # データ期間を動的に決定
    min_date = pivot_data.index.min()
    historical_data = pivot_data[pivot_data.index < '2025-06-01']
    predicted_data = pivot_data[pivot_data.index >= '2025-06-01']
    combined_data = pd.concat([historical_data, predicted_data])
    
    print(f'データ期間: {combined_data.index.min().strftime("%Y年%m月")} ～ {combined_data.index.max().strftime("%Y年%m月")}')
    print(f'実績データ数: {len(historical_data)}ヶ月')
    print(f'予測データ数: {len(predicted_data)}ヶ月')
    print(f'合計: {len(combined_data)}ヶ月')
    
    # グラフの設定
    plt.figure(figsize=(24, 14))
    
    # 年代別の色設定（はっきりした色）
    colors = {
        '10代未満': '#0070C0',  
        '10代': '#00C7A2',      
        '20代': '#FF00FF',      
        '30代': '#ED7D31',      
        '40代': '#FFC000',      
        '50代': '#006400',      
        '60代': '#7030A0',      
        '70代': '#00B0F0'       
    }
    
    # 年代順の列名
    age_order = ['10代未満', '10代', '20代', '30代', '40代', '50代', '60代', '70代']
    
    # x軸の位置
    x_pos = np.arange(len(combined_data))
    width = 0.8
    
    # 年代別に棒グラフを積み上げ
    bottom = np.zeros(len(combined_data))
    
    for age in age_order:
        if age in combined_data.columns:
            plt.bar(x_pos, combined_data[age], width, 
                   label=age, color=colors[age], 
                   bottom=bottom, alpha=0.9, edgecolor='white', linewidth=1)
            bottom += combined_data[age]
    
    # 実績と予測の境界線
    historical_end_idx = len(historical_data) - 1
    plt.axvline(x=historical_end_idx + 0.5, color='black', linestyle='--', linewidth=4, alpha=0.8)
    plt.text(historical_end_idx + 0.5, plt.ylim()[1]*0.95, '予測開始', 
            rotation=90, verticalalignment='top', fontsize=18, color='black', fontweight='bold')
    
    # グラフの装飾
    #plt.title('東康生通り１ 通行人数（年代別）の推移（月平均）', 
     #         fontsize=32, fontweight='bold', pad=40)
    
    plt.xlabel('年月', fontsize=20, fontweight='bold')
    plt.ylabel('月毎の年代別平均通行人数（人）', fontsize=20, fontweight='bold')
    
    # x軸のラベル設定（6ヶ月ごとに表示）
    tick_positions = range(0, len(combined_data), 6)
    tick_labels = [combined_data.index[i].strftime('%Y年%m月') for i in tick_positions if i < len(combined_data)]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    
    # 凡例設定
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
               fontsize=16, frameon=True, fancybox=True, shadow=True)
    
    # グリッド
    plt.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    filename = '東康生通り１_2020-2025_棒グラフ.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
if __name__ == "__main__":
    create_final_chart()
