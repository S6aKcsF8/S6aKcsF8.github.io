import subprocess
import sys
import os
import time
import glob
from datetime import datetime

def run_prediction_and_visualization():
    """予測とデータ作成を実行"""
    print(f"実行時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    
    try:
        # visualization_1_age_trend.pyを実行
        result = subprocess.run([sys.executable, 'visualization_1_age_trend.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ 予測システム実行エラー: {e}")
        return False
    
    return True

def run_chart_creation():
    """棒グラフ作成を実行"""
    try:
        # create_final_chart.pyを実行
        result = subprocess.run([sys.executable, 'create_final_chart.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ 棒グラフ作成エラー: {e}")
        return False
    
    return True

def update_dashboard_statistics():
    """ダッシュボードの統計情報を更新"""
    try:
        # 最新のCSVファイルを検索
        csv_files = glob.glob('age_trend_data_for_looker*.csv')
        if not csv_files:
            return False
        
        latest_file = max(csv_files, key=os.path.getctime)
        
        # データ読み込み
        import pandas as pd
        df = pd.read_csv(latest_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # 統計計算
        historical_data = df[df['data_type'] == 'historical']
        predicted_data = df[df['data_type'] == 'predicted']
        
        # 最新月の総通行人数
        latest_month = historical_data[historical_data['date'] == historical_data['date'].max()]
        latest_total = latest_month['value'].sum()
        
        # 予測期間平均通行人数
        pred_monthly = predicted_data.groupby('date')['value'].sum()
        pred_avg = pred_monthly.mean()
        
        # 変化率
        change_rate = ((pred_avg - latest_total) / latest_total) * 100
        
        # 期間情報
        historical_months = len(historical_data['date'].unique())
        predicted_months = len(predicted_data['date'].unique())
        total_months = historical_months + predicted_months
        
        latest_date = historical_data['date'].max().strftime('%Y年%m月')
        
        print(f"  - 最新月総通行人数: {latest_total:,.0f}人 ({latest_date})")
        print(f"  - 予測期間平均通行人数: {pred_avg:,.0f}人")
        print(f"  - 変化率: {change_rate:+.1f}%")
        print(f"  - データ期間: {total_months}ヶ月 (実績{historical_months}ヶ月 + 予測{predicted_months}ヶ月)")
        
        return {
            'latest_total': latest_total,
            'latest_date': latest_date,
            'pred_avg': pred_avg,
            'change_rate': change_rate,
            'total_months': total_months,
            'historical_months': historical_months,
            'predicted_months': predicted_months
        }
        
    except Exception as e:
        print(f"✗ 統計情報更新エラー: {e}")
        return False

def update_html_dashboard(stats):
    """HTMLダッシュボードを更新"""
    print("\n4. HTMLダッシュボードを更新")
    
    try:
        html_file = '東康生通り１_最終版_棒グラフダッシュボード.html'
        
        if not os.path.exists(html_file):
            print(f"✗ {html_file}が見つかりません")
            return False
        
        # HTMLファイルを読み込み
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # 統計情報を更新
        if stats:
            # 最新月総通行人数を更新
            html_content = html_content.replace(
                '<div class="stat-value">8,242人</div>',
                f'<div class="stat-value">{stats["latest_total"]:,.0f}人</div>'
            )
            
            # 最新月の期間を更新
            html_content = html_content.replace(
                '<div class="stat-period">2025年5月実績</div>',
                f'<div class="stat-period">{stats["latest_date"]}実績</div>'
            )
            
            # 予測期間平均通行人数を更新
            html_content = html_content.replace(
                '<div class="stat-value">11,178人</div>',
                f'<div class="stat-value">{stats["pred_avg"]:,.0f}人</div>'
            )
            
            # 変化率を更新
            html_content = html_content.replace(
                '<div class="stat-value">+35.6%</div>',
                f'<div class="stat-value">{stats["change_rate"]:+.1f}%</div>'
            )
            
            # データ期間を更新
            html_content = html_content.replace(
                '<div class="stat-value">66ヶ月</div>',
                f'<div class="stat-value">{stats["total_months"]}ヶ月</div>'
            )
            
            html_content = html_content.replace(
                '<div class="stat-period">実績54ヶ月 + 予測6ヶ月</div>',
                f'<div class="stat-period">実績{stats["historical_months"]}ヶ月 + 予測{stats["predicted_months"]}ヶ月</div>'
            )
        
        # 更新されたHTMLを保存
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True
        
    except Exception as e:
        print(f"✗ HTMLダッシュボード更新エラー: {e}")
        return False

def open_dashboard():
    """ダッシュボードを開く"""
    
    try:
        html_file = '東康生通り１_最終版_棒グラフダッシュボード.html'
        
        if os.path.exists(html_file):
            # Windowsでブラウザを開く
            os.startfile(html_file)
            print(f"✓ ダッシュボードをブラウザで開きました: {html_file}")
        else:
            print(f"✗ ダッシュボードファイルが見つかりません: {html_file}")
            
    except Exception as e:
        print(f"✗ ダッシュボードを開けませんでした: {e}")

def main():    
    start_time = time.time()
    
    # 1. 予測とデータ作成
    if not run_prediction_and_visualization():
        return
    
    # 少し待機
    time.sleep(2)
    
    # 2. 棒グラフ作成
    if not run_chart_creation():
        return
    
    # 3. 統計情報更新
    stats = update_dashboard_statistics()
    
    # 4. HTMLダッシュボード更新
    update_html_dashboard(stats)
    
    # 5. ダッシュボード表示
    open_dashboard()
    
    # 実行時間表示
    end_time = time.time()
    execution_time = end_time - start_time
    
  
    print(f"  実行時間: {execution_time:.1f}秒")
    print(f"  完了時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")

if __name__ == "__main__":
    main()
