"""
類似商圏 駅提案ツール - 東京都全駅版
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="類似商圏 駅提案ツール",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main-title { font-size:1.8rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem; }
  .sub-title { font-size:0.9rem; color:#666; margin-bottom:1.5rem; }
  .metric-card { background:#f8f9fa; border-radius:8px; padding:1rem;
    text-align:center; border:1px solid #e9ecef; }
  .metric-value { font-size:1.4rem; font-weight:700; color:#2d6a4f; }
  .metric-label { font-size:0.75rem; color:#888; margin-top:2px; }
  .station-tag { display:inline-block; background:#e8f5e9; color:#2d6a4f;
    border-radius:12px; padding:3px 10px; font-size:0.8rem; margin:2px; }
  .result-header { background:linear-gradient(135deg,#1a1a2e,#2d6a4f);
    color:white; padding:1rem 1.5rem; border-radius:8px; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# データ読み込み
# ============================================================
@st.cache_data
def load_data():
    """
    東京都全駅データを読み込む
    ※ GitHubに tokyo_stations.csv を配置して使用
    """
    try:
        df = pd.read_csv("tokyo_stations.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        # フォールバック: サンプルデータ
        df = pd.DataFrame([
            {"駅名":"吉祥寺","乗降客数":82000,"路線名":"JR中央線","事業者名":"JR","lat":35.7027,"lon":139.5796,"都道府県":"東京都"},
            {"駅名":"三鷹","乗降客数":61000,"路線名":"JR中央線","事業者名":"JR","lat":35.7026,"lon":139.5603,"都道府県":"東京都"},
            {"駅名":"阿佐ヶ谷","乗降客数":38000,"路線名":"JR中央線","事業者名":"JR","lat":35.7074,"lon":139.6365,"都道府県":"東京都"},
        ])
        st.warning("⚠️ tokyo_stations.csv が見つかりません。サンプルデータで動作しています。")

    # 地価（乗降客数から簡易推定 - 実運用では国交省APIに差し替え）
    df["地価推定"] = (df["乗降客数"] * 8 + 300000).clip(upper=1500000)
    # 飲食店数（乗降客数から簡易推定 - 実運用では経済センサスに差し替え）
    df["飲食店推定"] = (df["乗降客数"] / 250).astype(int).clip(lower=10)
    # 小売店推定
    df["小売店推定"] = (df["乗降客数"] / 200).astype(int).clip(lower=10)

    return df

FEATURE_COLS = ["乗降客数", "地価推定", "飲食店推定", "小売店推定"]

@st.cache_data
def run_clustering(df, n_clusters):
    X = df[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    # データ件数よりクラスタ数が多い場合は自動調整
    n_clusters = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df = df.copy()
    df["クラスタ"] = clusters.astype(str)
    df["PCA_X"] = X_pca[:, 0]
    df["PCA_Y"] = X_pca[:, 1]
    return df, pca.explained_variance_ratio_

def get_weights():
    try:
        from xgboost import XGBRegressor
        df = load_data()
        X = df[FEATURE_COLS].values.astype(float)
        y = (df["乗降客数"] * 0.5 + df["飲食店推定"] * 100) / 1000
        model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
        model.fit(X, y)
        return dict(zip(FEATURE_COLS, model.feature_importances_)), True
    except ImportError:
        return {col: 1.0/len(FEATURE_COLS) for col in FEATURE_COLS}, False

def find_similar(df, target, weights, top_n):
    X = df[FEATURE_COLS].values.copy().astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    w = np.array([weights.get(c, 1.0) for c in FEATURE_COLS])
    X_w = X_scaled * w
    sim = cosine_similarity(X_w)
    sim_df = pd.DataFrame(sim, index=df["駅名"], columns=df["駅名"])
    sims = sim_df[target].drop(target).sort_values(ascending=False).head(top_n)
    rows = []
    for st_name, score in sims.items():
        r = df[df["駅名"] == st_name].iloc[0]
        rows.append({
            "順位": len(rows) + 1,
            "駅名": st_name,
            "路線名": r["路線名"],
            "事業者": r["事業者名"],
            "類似度": round(score, 3),
            "乗降客数": f"{int(r['乗降客数']):,}人",
            "lat": r["lat"],
            "lon": r["lon"],
        })
    return pd.DataFrame(rows)

# ============================================================
# UI
# ============================================================
st.markdown('<div class="main-title">🏙️ 類似商圏 駅提案ツール</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">東京都 607駅対応 ｜ 駅名を入力すると商圏特性が似た駅を提案します</div>', unsafe_allow_html=True)

df_raw = load_data()
weights, xgb_used = get_weights()

with st.sidebar:
    st.markdown("### ⚙️ 検索設定")

    # 駅名テキスト入力（全駅対応）
    station_input = st.text_input(
        "🔍 基準となる駅名を入力",
        placeholder="例：吉祥寺、三鷹、阿佐ヶ谷...",
        value="吉祥寺"
    )

    # 候補サジェスト
    if station_input:
        matches = df_raw[df_raw["駅名"].str.contains(station_input, na=False)]["駅名"].tolist()
        if matches and station_input not in matches:
            st.caption(f"候補: {' / '.join(matches[:8])}")
            target = matches[0]
        elif station_input in df_raw["駅名"].values:
            target = station_input
        else:
            target = None
    else:
        target = None

    top_n = st.slider("表示件数", min_value=3, max_value=20, value=10)
    n_clusters = st.slider("商圏クラスタ数", min_value=3, max_value=10, value=6)

    st.markdown("---")
    if xgb_used:
        st.success("✅ XGBoost 重み使用中")
    else:
        st.info("ℹ️ 均等重み（XGBoost未インストール）")
    st.caption(f"登録駅数: {len(df_raw)}駅")
    st.caption("データ出典: 国土数値情報・オープンポータル")

# ============================================================
# 分析・表示
# ============================================================
if target is None:
    st.info("👈 左のサイドバーに駅名を入力してください")
    st.stop()

if target not in df_raw["駅名"].values:
    st.error(f"「{station_input}」はデータに含まれていません。別の駅名をお試しください。")
    st.stop()

with st.spinner("分析中..."):
    df_clustered, pca_ratio = run_clustering(df_raw, n_clusters)
    result_df = find_similar(df_raw, target, weights, top_n)

target_data = df_raw[df_raw["駅名"] == target].iloc[0]
target_cluster = df_clustered[df_clustered["駅名"] == target]["クラスタ"].values[0]
same_cluster = df_clustered[
    (df_clustered["クラスタ"] == target_cluster) &
    (df_clustered["駅名"] != target)
]["駅名"].tolist()

# ヘッダー
st.markdown(f"""
<div class="result-header">
  <b style="font-size:1.2rem">📍 {target}駅</b>　
  <span style="opacity:0.8">{target_data['路線名']} / {target_data['事業者名']}</span>
</div>
""", unsafe_allow_html=True)

# メトリクス
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{int(target_data["乗降客数"]):,}</div><div class="metric-label">1日乗降客数（人）</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">¥{int(target_data["地価推定"]):,}</div><div class="metric-label">地価推定（円/㎡）</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{int(target_data["飲食店推定"])}</div><div class="metric-label">飲食店推定数</div></div>', unsafe_allow_html=True)

st.markdown("")
if same_cluster:
    st.markdown(
        "**🏷️ 同じ商圏クラスタ：**　" +
        "".join([f'<span class="station-tag">{s}</span>' for s in same_cluster[:12]]),
        unsafe_allow_html=True
    )

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📋 類似駅リスト", "🗺️ 地図", "📊 商圏クラスタ図"])

with tab1:
    st.markdown(f"#### {target}駅に類似した駅 TOP {top_n}")
    display_df = result_df.drop(columns=["lat", "lon"])
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "類似度": st.column_config.ProgressColumn(
                "類似度", min_value=0, max_value=1, format="%.3f"
            )
        }
    )
    csv = display_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 CSVダウンロード",
        data=csv,
        file_name=f"類似駅_{target}.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown(f"#### 地図（{target}駅 と 類似駅TOP{top_n}）")
    map_data = [{"駅名": f"★{target}（基準）", "lat": target_data["lat"],
                 "lon": target_data["lon"], "種別": "基準駅", "路線名": target_data["路線名"]}]
    for _, row in result_df.iterrows():
        map_data.append({"駅名": row["駅名"], "lat": row["lat"], "lon": row["lon"],
                         "種別": "類似駅", "路線名": row["路線名"]})
    map_df = pd.DataFrame(map_data)
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="駅名",
        hover_data={"路線名": True, "lat": False, "lon": False},
        color="種別",
        color_discrete_map={"基準駅": "#e74c3c", "類似駅": "#2d6a4f"},
        size=[20 if t == "基準駅" else 12 for t in map_df["種別"]],
        zoom=10, height=500, mapbox_style="open-street-map"
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.markdown(f"#### 商圏クラスタ図（PCA 2次元）")
    st.caption(f"PC1寄与率: {pca_ratio[0]:.1%}　PC2寄与率: {pca_ratio[1]:.1%}")
    df_plot = df_clustered.copy()
    df_plot["サイズ"] = df_plot["駅名"].apply(lambda x: 20 if x == target else 8)
    fig_pca = px.scatter(
        df_plot, x="PCA_X", y="PCA_Y", color="クラスタ",
        hover_name="駅名",
        hover_data={"乗降客数": True, "路線名": True, "PCA_X": False, "PCA_Y": False, "サイズ": False},
        size="サイズ", height=500,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_pca.update_layout(
        xaxis_title="第1主成分（商圏規模）",
        yaxis_title="第2主成分（商圏特性）"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("---")
st.caption("📌 データ: 国土数値情報・オープンポータル（東京都607駅）｜地価・飲食店数は乗降客数からの推定値")
