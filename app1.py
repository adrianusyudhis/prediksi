import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def extract_ranking(value):
    match = re.search(r'\((\d+)\.', str(value))
    return int(match.group(1)) if match else 21

def map_team_ranking_kategori(rank):
    if rank <= 5:
        return 0
    elif rank <= 12:
        return 1
    else:
        return 2

st.title("Prediksi Hasil Pertandingan Real Madrid")

uploaded_file = st.file_uploader("Upload dataset pertandingan Real Madrid (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['Date'] = pd.to_datetime(df['Date_Converted'])
        df = df.sort_values('Date') 

        df['Home_Ranking'] = df['Home_Ranking'].fillna('Tidak Diketahui (21)')
        df['Away_Ranking'] = df['Away_Ranking'].fillna('Tidak Diketahui (21)')

        team_mapping = dict(zip(df['Team_Name'], df['Team']))
        df['Win_Streak_Home'] = df.apply(lambda r: 1 if r['Results'] == 0 and r['HomeAway'] == 0 else 0, axis=1)
        df['Win_Streak_Away'] = df.apply(lambda r: 1 if r['Results'] == 0 and r['HomeAway'] == 1 else 0, axis=1)
        df['Clean_Sheet'] = df['GetGoal'].apply(lambda x: 1 if x == 0 else 0)
        df['Clean_Sheet_Last3'] = df['Clean_Sheet'].rolling(3).sum().shift(1).fillna(0)

        df['Madrid_Ranking_Num'] = df.apply(
            lambda row: extract_ranking(row['Home_Ranking']) if row['HomeAway'] == 0
            else extract_ranking(row['Away_Ranking']),
            axis=1
        )
        df['TeamRanking'] = df.apply(
            lambda row: map_team_ranking_kategori(
                extract_ranking(row['Away_Ranking']) if row['HomeAway'] == 0
                else extract_ranking(row['Home_Ranking'])
            ),
            axis=1
        )

        df['H2H_Matches'], df['H2H_Wins'] = 0, 0
        for idx, row in df.iterrows():
            h2h = df[(df['Team_Name'] == row['Team_Name']) & (df['Date'] < row['Date'])]
            df.at[idx, 'H2H_Matches'] = len(h2h)
            df.at[idx, 'H2H_Wins'] = len(h2h[h2h['Results'] == 0])
        df['H2H_Win_Rate'] = df.apply(lambda r: r['H2H_Wins'] / r['H2H_Matches'] if r['H2H_Matches'] > 0 else 0, axis=1)

        df['Goal_Diff'] = df['ScoreGoal'] - df['GetGoal']
        df['Avg_Goal_Diff_Last5'] = df['Goal_Diff'].rolling(5).mean().shift(1).fillna(0)

        df['Recent_7Days_Matches'] = 0
        for idx, row in df.iterrows():
            start_date = row['Date'] - timedelta(days=7)
            df.at[idx, 'Recent_7Days_Matches'] = len(df[(df['Date'] >= start_date) & (df['Date'] < row['Date'])])

        df['Opponent_Goal_Avg_Last5'] = df.groupby('Team_Name')['ScoreGoal'].transform(lambda x: x.shift(1).rolling(5).mean()).fillna(0)
        df['RM_Goal_Concede_Avg_Last5'] = df['GetGoal'].rolling(5).mean().shift(1).fillna(0)

        df['Days_Since_Last'] = df['Date'].diff().dt.days.fillna(7)

        features = [
            'MathType', 'HomeAway', 'Team', 'TeamRanking', 'Madrid_Ranking_Num',
            'Clean_Sheet_Last3', 'H2H_Matches', 'H2H_Wins', 'H2H_Win_Rate',
            'Avg_Goal_Diff_Last5', 'Recent_7Days_Matches',
            'Opponent_Goal_Avg_Last5', 'RM_Goal_Concede_Avg_Last5', 'Days_Since_Last'
        ]
        target = 'Results'

        df_model = df[features + [target]].copy().fillna(0)
        X = pd.get_dummies(df_model[features])
        y = df_model[target]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, stratify=y_resampled, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                              random_state=42, eval_metric='mlogloss', use_label_encoder=False)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        st.markdown("### Evaluasi Model")
        st.write(f"**Akurasi:** {acc:.2%}")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

        st.markdown("### Prediksi Pertandingan Baru")
        match_type = st.selectbox("Tipe Pertandingan", [0, 1, 2], format_func=lambda x: ["UCL", "La Liga", "Copa"][x])
        home_away = st.selectbox("Kandang / Tandang", [0, 1], format_func=lambda x: "Kandang" if x == 0 else "Tandang")
        team_name = st.selectbox("Nama Tim Lawan", sorted(df['Team_Name'].unique()))
        team_ranking_text = st.selectbox("Papan Lawan", ["Papan Atas (1-5)", "Papan Tengah (6-12)", "Papan Bawah (13-21)"])
        madrid_ranking_num = st.number_input("Ranking Real Madrid", 1, 100, 10)
        clean_sheet_last3 = st.slider("Clean Sheet (3 laga)", 0, 3, 0)
        avg_goal_diff_last5 = st.number_input("Rata-rata Selisih Gol", -5.0, 5.0, 0.0, step=0.1)
        recent_7days = st.slider("Jumlah Laga 7 Hari", 0, 5, 0)
        opponent_goal_avg = st.number_input("Rata-rata Gol Lawan (5 laga)", 0.0, 5.0, 1.0, step=0.1)
        rm_concede_avg = st.number_input("Rata-rata Kebobolan RM (5 laga)", 0.0, 5.0, 1.0, step=0.1)
        days_since_last = st.slider("Jumlah Jeda Hari Sejak Laga Terakhir", 0, 14, 7)

        team_ranking = {"Papan Atas (1-5)": 0, "Papan Tengah (6-12)": 1, "Papan Bawah (13-21)": 2}[team_ranking_text]
        team = team_mapping.get(team_name, 0)
        h2h_matches = len(df[df['Team_Name'] == team_name])
        h2h_wins = len(df[(df['Team_Name'] == team_name) & (df['Results'] == 0)])
        h2h_win_rate = h2h_wins / h2h_matches if h2h_matches > 0 else 0

        df_filtered = df[(df['Team_Name'] == team_name) & (df['MathType'] == match_type)]

        if st.button("Prediksi Hasil"):
            if df_filtered.empty:
                st.warning("⚠️ Tidak ada riwayat pertandingan melawan tim ini untuk tipe pertandingan yang dipilih. Prediksi tidak dapat dilakukan.")
            else:
                input_dict = {
                    'MathType': match_type,
                    'HomeAway': home_away,
                    'Team': team,
                    'TeamRanking': team_ranking,
                    'Madrid_Ranking_Num': madrid_ranking_num,
                    'Clean_Sheet_Last3': clean_sheet_last3,
                    'H2H_Matches': h2h_matches,
                    'H2H_Wins': h2h_wins,
                    'H2H_Win_Rate': h2h_win_rate,
                    'Avg_Goal_Diff_Last5': avg_goal_diff_last5,
                    'Recent_7Days_Matches': recent_7days,
                    'Opponent_Goal_Avg_Last5': opponent_goal_avg,
                    'RM_Goal_Concede_Avg_Last5': rm_concede_avg,
                    'Days_Since_Last': days_since_last
                }

                input_df = pd.DataFrame([input_dict])
                input_encoded = pd.get_dummies(input_df)

                for col in X.columns:
                    if col not in input_encoded:
                        input_encoded[col] = 0
                input_encoded = input_encoded[X.columns]

                input_scaled = scaler.transform(input_encoded)
                pred = model.predict(input_scaled)[0]
                st.subheader(f"📢 Prediksi Hasil: **{['Menang', 'Imbang', 'Kalah'][pred]}**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
