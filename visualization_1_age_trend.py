import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """データの読み込みと前処理"""
    try:
        df = pd.read_csv(filepath)
        print(f"データ形状: {df.shape}")
        print(f"カラム: {df.columns.tolist()}")
        
        # 日付列の確認と変換
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print(f"期間: {df['date'].min()} ～ {df['date'].max()}")
        else:
            print("警告: 'date'カラムが見つかりません")
            print(f"利用可能なカラム: {df.columns.tolist()}")
            return None, None, None
        
        # 月次リサンプリング（合計値）
        df_ts = df.set_index("date").resample("M").sum()
        
        # 年代別カラムの確認と抽出
        cols = ['Age00', 'Age10', 'Age20', 'Age30', 'Age40', 'Age50', 'Age60', 'Age70']
        available_cols = [col for col in cols if col in df_ts.columns]
        
        if not available_cols:
            print("警告: 年代別カラムが見つかりません。")
            print(f"利用可能なカラム: {df_ts.columns.tolist()}")
            # 代替カラム名の検索
            age_related_cols = [col for col in df_ts.columns if any(age in col.lower() for age in ['age', '年代', '10代', '20代', '30代', '40代', '50代', '60代', '70代'])]
            if age_related_cols:
                print(f"年代関連カラム候補: {age_related_cols}")
                available_cols = age_related_cols[:8]  # 最大8個まで
            else:
                # 数値カラムを年代として仮定
                numeric_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    available_cols = numeric_cols[:8]
                    print(f"数値カラムを年代データとして使用: {available_cols}")
                else:
                    return None, None, None
        
        # 欠損値の補間
        data = df_ts[available_cols].ffill().bfill()
        print(f"使用するカラム: {available_cols}")
        print(f"処理後データ形状: {data.shape}")
        
        return data, available_cols, df_ts
    
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None, None, None

def create_supervised_dataset(data):
    """時系列データを教師あり学習用に変換（全データを使用）"""
    X, y = [], []
    sequence_length = len(data)
    
    # 各時点でそれまでの全データを使用
    for i in range(1, sequence_length):
        X.append(data[:i].flatten())
        y.append(data[i])
    
    # パディングを適用して入力の長さを揃える
    max_length = data[:sequence_length-1].flatten().shape[0]
    X_padded = []
    for x in X:
        padding = np.zeros(max_length - len(x))
        X_padded.append(np.concatenate([padding, x]))
    
    return np.array(X_padded), np.array(y)

def build_and_train_models(X_train, X_test, y_train, y_test, scaler):
    """複数モデルの構築・学習・評価"""
    models = {}
    results = {}
    
    # MLPRegressor（深層学習）
    print("\n=== MLPRegressorモデル（深層学習） ===")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_model.fit(X_train, y_train)
    mlp_pred = mlp_model.predict(X_test)
    mlp_pred_original = scaler.inverse_transform(mlp_pred)
    y_test_original = scaler.inverse_transform(y_test)
    mlp_mse = mean_squared_error(y_test_original, mlp_pred_original)
    
    models['MLP'] = mlp_model
    results['MLP'] = {
        'MSE': mlp_mse,
        'MAE': mean_absolute_error(y_test_original, mlp_pred_original),
        'RMSE': np.sqrt(mlp_mse),
        'predictions': mlp_pred_original
    }
    
    # RandomForest（機械学習）
    print("\n=== RandomForestモデル（機械学習） ===")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,  # より深い木を許可
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_original = scaler.inverse_transform(rf_pred)
    rf_mse = mean_squared_error(y_test_original, rf_pred_original)
    
    models['RandomForest'] = rf_model
    results['RandomForest'] = {
        'MSE': rf_mse,
        'MAE': mean_absolute_error(y_test_original, rf_pred_original),
        'RMSE': np.sqrt(rf_mse),
        'predictions': rf_pred_original
    }
    
    # 最適モデル選択
    best_model_name = 'MLP' if mlp_mse <= rf_mse else 'RandomForest'
    best_model = models[best_model_name]
    
    print(f"\n=== モデル評価結果 ===")
    for name, result in results.items():
        print(f"{name}: MSE={result['MSE']:.4f}, MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}")
    print(f"最適モデル: {best_model_name}")
    
    return best_model, best_model_name, results

def predict_future(model, data_scaled, scaler, n_months=6):
    """未来予測（全データを使用）"""
    future_predictions = []
    current_data = data_scaled.copy()
    
    for i in range(n_months):
        # 全データを使用して予測
        input_sequence = current_data.flatten()
        # パディングを適用（最新のモデルの入力サイズに合わせる）
        padding_size = model.n_features_in_ - len(input_sequence)
        if padding_size > 0:
            input_sequence = np.concatenate([np.zeros(padding_size), input_sequence])
        elif padding_size < 0:
            input_sequence = input_sequence[-model.n_features_in_:]
            
        input_sequence = input_sequence.reshape(1, -1)
        pred = model.predict(input_sequence)
        future_predictions.append(pred[0])
        current_data = np.vstack([current_data, pred])
    
    predicted_values = scaler.inverse_transform(future_predictions)
    return predicted_values

def get_age_label(age_col):
    """年代ラベルの変換"""
    labels = {
        'Age00': '10代未満',
        'Age10': '10代',
        'Age20': '20代',
        'Age30': '30代',
        'Age40': '40代',
        'Age50': '50代',
        'Age60': '60代',
        'Age70': '70代'
    }
    return labels.get(age_col, age_col)

def create_age_trend_visualization(data, predictions, cols, model_results):
    """年代別通行人数の推移と予測の可視化"""
    plt.figure(figsize=(16, 10))
    
    # カラーパレット
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    age_labels = [get_age_label(col) for col in cols]
    
    # 歴史データ
    for i, col in enumerate(cols):
        plt.plot(data.index, data[col], color=colors[i % len(colors)], 
                label=f'{age_labels[i]} (実績)', linewidth=2.5, alpha=0.8)
    
    # 予測データ
    future_dates = pd.date_range(
        start=data.index[-1] + pd.offsets.MonthBegin(),
        periods=len(predictions),
        freq='MS'
    )
    
    for i, col in enumerate(cols):
        plt.plot(future_dates, predictions[:, i], 
                color=colors[i % len(colors)], linestyle='--', 
                label=f'{age_labels[i]} (予測)', linewidth=2.5, alpha=0.8)
    
    # 予測開始線
    plt.axvline(x=data.index[-1], color='red', linestyle=':', alpha=0.7, linewidth=3)
    plt.text(data.index[-1], plt.ylim()[1]*0.9, '予測開始', 
            rotation=90, verticalalignment='top', fontsize=14, color='red', weight='bold')
    
    plt.title('東康生通り１_通行人数（年代別）の推移と予測', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('年月', fontsize=14, fontweight='bold')
    plt.ylabel('月間通行人数（人）', fontsize=14, fontweight='bold')
    
    # 凡例を2列で表示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # レイアウト調整
    plt.tight_layout()
    
    # ファイル保存
    #plt.savefig('visualization_1_age_trend.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_looker_studio_data(data, predictions, cols):
    """Looker Studio用データの作成"""
    # 歴史データの準備
    historical_data = data.copy()
    historical_data['data_type'] = 'historical'
    
    # 予測データの準備
    future_dates = pd.date_range(
        start=data.index[-1] + pd.offsets.MonthBegin(),
        periods=len(predictions),
        freq='MS'
    )
    
    pred_df = pd.DataFrame(predictions, columns=cols, index=future_dates)
    pred_df['data_type'] = 'predicted'
    
    # 結合
    combined_data = pd.concat([historical_data[cols + ['data_type']], pred_df])
    
    # Looker Studio用のフォーマット（縦持ち）
    dashboard_data = []
    for date, row in combined_data.iterrows():
        for age_group in cols:
            dashboard_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'age_group': age_group,
                'age_label': get_age_label(age_group),
                'value': row[age_group],
                'data_type': row['data_type']
            })
    
    dashboard_df = pd.DataFrame(dashboard_data)
    
    # CSV出力（エラーハンドリング付き）
    import time
    import os
    
    try:
        # 既存ファイルが開いている場合があるため、少し待機
        time.sleep(1)
        
        # ファイルが使用中の場合、別名で保存
        looker_filename = 'age_trend_data_for_looker.csv'
        combined_filename = 'age_trend_combined_data.csv'
        
        try:
            dashboard_df.to_csv(looker_filename, index=False, encoding='utf-8-sig')
        except PermissionError:
            timestamp = int(time.time())
            looker_filename = f'age_trend_data_for_looker_{timestamp}.csv'
            dashboard_df.to_csv(looker_filename, index=False, encoding='utf-8-sig')
            print(f"警告: 元のファイルが使用中のため、{looker_filename}として保存しました")
        
        try:
            combined_data.to_csv(combined_filename, encoding='utf-8-sig')
        except PermissionError:
            timestamp = int(time.time())
            combined_filename = f'age_trend_combined_data_{timestamp}.csv'
            combined_data.to_csv(combined_filename, encoding='utf-8-sig')
            print(f"警告: 元のファイルが使用中のため、{combined_filename}として保存しました")
        
        print("\n=== Looker Studio用データファイル作成完了 ===")
        print(f"1. {looker_filename} - Looker Studio用データ（縦持ち）")
        print(f"2. {combined_filename} - 歴史+予測データ（横持ち）")
        
    except Exception as e:
        print(f"CSVファイル保存エラー: {e}")
        print("データは正常に処理されましたが、ファイル保存に失敗しました")
    
    return dashboard_df, combined_data

def main():
    """メイン処理 - 年代別通行人数の推移と予測"""
    print("=== 可視化1: 年代別通行人数の推移と予測 ===")
    
    filepath = "camera_0.csv"
    data, cols, df_original = load_and_preprocess_data(filepath)
    
    if data is None:
        print("データの読み込みに失敗しました。")
        return
    
    # データの正規化
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 時系列データセット作成（全データを使用）
    X, y = create_supervised_dataset(data_scaled)
    
    if len(X) == 0:
        print("学習データが不足しています。")
        return
    
    # 訓練・テストデータ分割（最新の6ヶ月をテストデータとして使用）
    test_size = min(6 / len(X), 0.2)  # 最大でも20%をテストデータとして使用
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=42
    )
    
    print(f"訓練データサイズ: {len(X_train)}, テストデータサイズ: {len(X_test)}")
    
    # モデル学習・評価
    best_model, best_model_name, model_results = build_and_train_models(
        X_train, X_test, y_train, y_test, scaler
    )
    
    # 未来予測（6ヶ月）
    print("\n=== 未来予測実行中 ===")
    future_predictions = predict_future(
        best_model, data_scaled, scaler, n_months=6
    )
    
    # 可視化
    print("\n=== 年代別推移可視化作成中 ===")
    create_age_trend_visualization(data, future_predictions, cols, model_results)
    
    # Looker Studio用データ作成
    print("\n=== Looker Studio用データ作成中 ===")
    dashboard_df, combined_data = create_looker_studio_data(data, future_predictions, cols)
    
    # インサイト生成
    print("\n=== 分析結果インサイト ===")
    latest_month = data.iloc[-1]
    total_current = latest_month.sum()
    pred_avg = np.mean(future_predictions, axis=0)
    total_pred = pred_avg.sum()
    
    print(f"最新月総通行人数: {total_current:,.0f}人")
    print(f"予測期間平均総通行人数: {total_pred:,.0f}人")
    change_rate = ((total_pred - total_current) / total_current) * 100
    print(f"変化率: {change_rate:+.1f}%")
    
    print(f"\n=== 処理完了 ===")
    print(f"使用モデル: {best_model_name}")
    print("年代別通行人数の推移と予測グラフとデータが生成されました。")

if __name__ == "__main__":
    main()
