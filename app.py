"""
類似商圏・類似駅提案ツール
不動産スタッフ向け Streamlit アプリ
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="類似商圏 駅提案ツール",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# スタイル
# ============================================================
st.markdown("""
<style>
  .main-title {
    font-size: 1.8rem; font-weight: 700;
    color: #1a1a2e; margin-bottom: 0.2rem;
  }
  .sub-title {
    font-size: 0.9rem; color: #666; margin-bottom: 1.5rem;
  }
  .metric-card {
    background: #f8f9fa; border-radius: 8px;
    padding: 1rem; text-align: center;
    border: 1px solid #e9ecef;
  }
  .metric-value {
    font-size: 1.4rem; font-weight: 700; color: #2d6a4f;
  }
  .metric-label {
    font-size: 0.75rem; color: #888; margin-top: 2px;
  }
  .station-tag {
    display: inline-block;
    background: #e8f5e9; color: #2d6a4f;
    border-radius: 12px; padding: 3px 10px;
    font-size: 0.8rem; margin: 2px;
  }
  .result-header {
    background: linear-gradient(135deg, #1a1a2e, #2d6a4f);
    color: white; padding: 1rem 1.5rem;
    border-radius: 8px; margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ============================================================
# データ定義（サンプル）
# 実運用では国交省・e-Stat APIから取得
# ============================================================
@st.cache_data
def load_station_data():
    stations = [
        {"駅名": "吉祥寺",   "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 82000,  "地価": 850000,  "1km圏人口": 42000, "飲食店数": 380, "小売店数": 520, "従業者数": 4800, "lat": 35.7027, "lon": 139.5796},
        {"駅名": "三鷹",     "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 61000,  "地価": 520000,  "1km圏人口": 38000, "飲食店数": 210, "小売店数": 280, "従業者数": 2600, "lat": 35.7026, "lon": 139.5603},
        {"駅名": "阿佐ヶ谷", "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 38000,  "地価": 480000,  "1km圏人口": 35000, "飲食店数": 180, "小売店数": 220, "従業者数": 1900, "lat": 35.7074, "lon": 139.6365},
        {"駅名": "高円寺",   "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 52000,  "地価": 510000,  "1km圏人口": 41000, "飲食店数": 240, "小売店数": 310, "従業者数": 2800, "lat": 35.7055, "lon": 139.6493},
        {"駅名": "荻窪",     "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 72000,  "地価": 580000,  "1km圏人口": 39000, "飲食店数": 280, "小売店数": 360, "従業者数": 3200, "lat": 35.7058, "lon": 139.6200},
        {"駅名": "西荻窪",   "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 31000,  "地価": 430000,  "1km圏人口": 33000, "飲食店数": 140, "小売店数": 170, "従業者数": 1500, "lat": 35.7046, "lon": 139.5985},
        {"駅名": "中野",     "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 98000,  "地価": 720000,  "1km圏人口": 48000, "飲食店数": 350, "小売店数": 480, "従業者数": 4200, "lat": 35.7076, "lon": 139.6654},
        {"駅名": "下北沢",   "路線": "京王井の頭線",  "都道府県": "東京都", "乗降客数": 69000,  "地価": 620000,  "1km圏人口": 37000, "飲食店数": 290, "小売店数": 380, "従業者数": 3500, "lat": 35.6613, "lon": 139.6681},
        {"駅名": "自由が丘", "路線": "東急東横線",    "都道府県": "東京都", "乗降客数": 74000,  "地価": 780000,  "1km圏人口": 36000, "飲食店数": 260, "小売店数": 420, "従業者数": 3800, "lat": 35.6079, "lon": 139.6681},
        {"駅名": "二子玉川", "路線": "東急田園都市線","都道府県": "東京都", "乗降客数": 88000,  "地価": 820000,  "1km圏人口": 34000, "飲食店数": 180, "小売店数": 350, "従業者数": 3200, "lat": 35.6083, "lon": 139.6275},
        {"駅名": "恵比寿",   "路線": "JR山手線",      "都道府県": "東京都", "乗降客数": 105000, "地価": 1200000, "1km圏人口": 31000, "飲食店数": 420, "小売店数": 380, "従業者数": 5200, "lat": 35.6467, "lon": 139.7101},
        {"駅名": "代官山",   "路線": "東急東横線",    "都道府県": "東京都", "乗降客数": 28000,  "地価": 950000,  "1km圏人口": 28000, "飲食店数": 180, "小売店数": 250, "従業者数": 2400, "lat": 35.6488, "lon": 139.7026},
        {"駅名": "中目黒",   "路線": "東急東横線",    "都道府県": "東京都", "乗降客数": 91000,  "地価": 890000,  "1km圏人口": 33000, "飲食店数": 320, "小売店数": 290, "従業者数": 3800, "lat": 35.6441, "lon": 139.6984},
        {"駅名": "学芸大学", "路線": "東急東横線",    "都道府県": "東京都", "乗降客数": 43000,  "地価": 620000,  "1km圏人口": 35000, "飲食店数": 160, "小売店数": 190, "従業者数": 1700, "lat": 35.6275, "lon": 139.6827},
        {"駅名": "祐天寺",   "路線": "東急東横線",    "都道府県": "東京都", "乗降客数": 29000,  "地価": 520000,  "1km圏人口": 32000, "飲食店数": 100, "小売店数": 120, "従業者数": 1100, "lat": 35.6341, "lon": 139.6870},
        {"駅名": "三軒茶屋", "路線": "東急田園都市線","都道府県": "東京都", "乗降客数": 88000,  "地価": 680000,  "1km圏人口": 40000, "飲食店数": 310, "小売店数": 340, "従業者数": 3600, "lat": 35.6435, "lon": 139.6694},
        {"駅名": "武蔵小山", "路線": "東急目黒線",    "都道府県": "東京都", "乗降客数": 52000,  "地価": 540000,  "1km圏人口": 36000, "飲食店数": 170, "小売店数": 230, "従業者数": 2100, "lat": 35.6218, "lon": 139.7161},
        {"駅名": "西小山",   "路線": "東急目黒線",    "都道府県": "東京都", "乗降客数": 24000,  "地価": 420000,  "1km圏人口": 31000, "飲食店数": 90,  "小売店数": 110, "従業者数": 980,  "lat": 35.6175, "lon": 139.7059},
        {"駅名": "戸越銀座", "路線": "東急池上線",    "都道府県": "東京都", "乗降客数": 22000,  "地価": 380000,  "1km圏人口": 34000, "飲食店数": 130, "小売店数": 200, "従業者数": 1600, "lat": 35.6035, "lon": 139.7192},
        {"駅名": "蒲田",     "路線": "JR京浜東北線",  "都道府県": "東京都", "乗降客数": 110000, "地価": 410000,  "1km圏人口": 46000, "飲食店数": 380, "小売店数": 490, "従業者数": 4500, "lat": 35.5637, "lon": 139.7160},
        {"駅名": "立川",     "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 145000, "地価": 580000,  "1km圏人口": 44000, "飲食店数": 420, "小売店数": 580, "従業者数": 5800, "lat": 35.6981, "lon": 139.4131},
        {"駅名": "国分寺",   "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 95000,  "地価": 480000,  "1km圏人口": 38000, "飲食店数": 250, "小売店数": 330, "従業者数": 3100, "lat": 35.7024, "lon": 139.4778},
        {"駅名": "国立",     "路線": "JR中央線",      "都道府県": "東京都", "乗降客数": 38000,  "地価": 420000,  "1km圏人口": 32000, "飲食店数": 120, "小売店数": 160, "従業者数": 1400, "lat": 35.6877, "lon": 139.4454},
        {"駅名": "町田",     "路線": "JR横浜線",      "都道府県": "東京都", "乗降客数": 140000, "地価": 450000,  "1km圏人口": 43000, "飲食店数": 450, "小売店数": 620, "従業者数": 6200, "lat": 35.5422, "lon": 139.4458},
        {"駅名": "調布",     "路線": "京王線",        "都道府県": "東京都", "乗降客数": 92000,  "地価": 450000,  "1km圏人口": 37000, "飲食店数": 280, "小売店数": 360, "従業者数": 3400, "lat": 35.6518, "lon": 139.5455},
        {"駅名": "府中",     "路線": "京王線",        "都道府県": "東京都", "乗降客数": 88000,  "地価": 420000,  "1km圏人口": 36000, "飲食店数": 260, "小売店数": 340, "従業者数": 3200, "lat": 35.6699, "lon": 139.4777},
        {"駅名": "成城学園前","路線": "小田急小田原線","都道府県": "東京都", "乗降客数": 72000,  "地価": 610000,  "1km圏人口": 33000, "飲食店数": 140, "小売店数": 200, "従業者数": 1900, "lat": 35.6318, "lon": 139.6052},
        {"駅名": "登戸",     "路線": "小田急小田原線","都道府県": "東京都", "乗降客数": 89000,  "地価": 380000,  "1km圏人口": 38000, "飲食店数": 160, "小売店数": 210, "従業者数": 1800, "lat": 35.6152, "lon": 139.5872},
        {"駅名": "新百合ヶ丘","路線": "小田急小田原線","都道府県": "神奈川県","乗降客数": 103000, "地価": 420000,  "1km圏人口": 36000, "飲食店数": 200, "小売店数": 290, "従業者数": 2700, "lat": 35.6000, "lon": 139.5004},
        {"駅名": "溝の口",   "路線": "東急田園都市線","都道府県": "神奈川県","乗降客数": 118000, "地価": 480000,  "1km圏人口": 42000, "飲食店数": 290, "小売店数": 380, "従業者数": 3500, "lat": 35.6017, "lon": 139.6095},
        {"駅名": "武蔵溝ノ口","路線": "JR南武線",     "都道府県": "神奈川県","乗降客数": 75000,  "地価": 460000,  "1km圏人口": 38000, "飲食店数": 160, "小売店数": 210, "従業者数": 1900, "lat": 35.6017, "lon": 139.6095},
    ]
    return pd.DataFrame(stations)


# ============================================================
# 分析ロジック
# ============================================================
FEATURE_COLS = ["乗降客数", "地価", "1km圏人口", "飲食店数", "小売店数", "従業者数"]

@st.cache_data
def build_features(df):
    """派生特徴量を追加"""
    df = df.copy()
    df["飲食店密度"] = df["飲食店数"] / df["1km圏人口"] * 10000
    df["小売店密度"] = df["小売店数"] / df["1km圏人口"] * 10000
    return df

@st.cache_data
def run_clustering(df, n_clusters=6):
    """PCA + K-meansクラスタリング"""
    cols = FEATURE_COLS + ["飲食店密度", "小売店密度"]
    X = df[cols].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["クラスタ"] = clusters.astype(str)
    df["PCA_X"] = X_pca[:, 0]
    df["PCA_Y"] = X_pca[:, 1]
    return df, pca.explained_variance_ratio_

def get_weights():
    """
    XGBoostが使える場合はXGBoostの重み、
    なければ均等重みを返す
    """
    try:
        from xgboost import XGBRegressor
        df = load_station_data()
        df = build_features(df)
        cols = FEATURE_COLS + ["飲食店密度", "小売店密度"]
        X = df[cols].values.astype(float)
        y = (df["乗降客数"] * 0.4 + df["飲食店数"] * 100 + df["小売店数"] * 80) / 1000
        model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
        model.fit(X, y)
        return dict(zip(cols, model.feature_importances_)), cols, True
    except ImportError:
        cols = FEATURE_COLS + ["飲食店密度", "小売店密度"]
        return {col: 1.0/len(cols) for col in cols}, cols, False

def find_similar(df, target, weights, feature_cols, top_n):
    """重み付きコサイン類似度で類似駅を算出"""
    X = df[feature_cols].values.copy().astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    w = np.array([weights.get(c, 1.0) for c in feature_cols])
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
            "路線": r["路線"],
            "類似度": round(score, 3),
            "乗降客数": f"{int(r['乗降客数']):,}人",
            "地価(円/㎡)": f"¥{int(r['地価']):,}",
            "1km圏人口": f"{int(r['1km圏人口']):,}人",
            "飲食店数": int(r["飲食店数"]),
            "小売店数": int(r["小売店数"]),
            "lat": r["lat"],
            "lon": r["lon"],
        })
    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================
# ヘッダー
st.markdown('<div class="main-title">🏙️ 類似商圏 駅提案ツール</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">駅名を入力すると、商圏特性が似た駅をAIが提案します</div>', unsafe_allow_html=True)

# サイドバー
with st.sidebar:
    st.markdown("### ⚙️ 検索設定")
    df_raw = load_station_data()
    df_feat = build_features(df_raw)

    station_list = sorted(df_feat["駅名"].tolist())
    target = st.selectbox("🔍 基準となる駅を選択", station_list, index=station_list.index("吉祥寺"))
    top_n = st.slider("表示件数", min_value=3, max_value=15, value=8)
    n_clusters = st.slider("商圏クラスタ数", min_value=3, max_value=8, value=6)

    st.markdown("---")
    st.markdown("### 📊 重み付け方法")
    weights, feature_cols, xgb_used = get_weights()
    if xgb_used:
        st.success("✅ XGBoost 重み使用中")
    else:
        st.info("ℹ️ 均等重み（XGBoost未インストール）")

    st.markdown("---")
    st.markdown("### 📋 データについて")
    st.caption("乗降客数・地価・人口・経済センサスをもとに商圏類似度を算出")
    st.caption(f"登録駅数: {len(df_raw)}駅")

# ============================================================
# 分析実行
# ============================================================
with st.spinner("分析中..."):
    df_clustered, pca_ratio = run_clustering(df_feat, n_clusters)
    result_df = find_similar(df_feat, target, weights, feature_cols, top_n)

# 対象駅のクラスタ
target_cluster = df_clustered[df_clustered["駅名"] == target]["クラスタ"].values[0]
same_cluster = df_clustered[
    (df_clustered["クラスタ"] == target_cluster) &
    (df_clustered["駅名"] != target)
]["駅名"].tolist()

target_data = df_feat[df_feat["駅名"] == target].iloc[0]

# ============================================================
# 結果表示
# ============================================================

# 対象駅の概要
st.markdown(f"""
<div class="result-header">
  <b style="font-size:1.2rem">📍 {target}駅</b>　
  <span style="opacity:0.8">{target_data['路線']} / {target_data['都道府県']}</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{int(target_data["乗降客数"]):,}</div><div class="metric-label">1日乗降客数（人）</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">¥{int(target_data["地価"]):,}</div><div class="metric-label">地価（円/㎡）</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{int(target_data["1km圏人口"]):,}</div><div class="metric-label">1km圏内人口（人）</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{int(target_data["飲食店数"])}</div><div class="metric-label">飲食店数</div></div>', unsafe_allow_html=True)

st.markdown("")

# 同クラスタ
st.markdown("**🏷️ 同じ商圏クラスタの駅：**　" + "　".join([f'<span class="station-tag">{s}</span>' for s in same_cluster]), unsafe_allow_html=True)

st.markdown("---")

# タブで結果・地図・グラフを切り替え
tab1, tab2, tab3 = st.tabs(["📋 類似駅リスト", "🗺️ 地図", "📊 商圏クラスタ図"])

with tab1:
    st.markdown(f"#### {target}駅に類似した駅 TOP {top_n}")
    display_df = result_df.drop(columns=["lat", "lon"])

    # 類似度でカラーバー表示
    st.dataframe(
        display_df.style.background_gradient(subset=["類似度"], cmap="Greens"),
        use_container_width=True,
        hide_index=True
    )

    # CSVダウンロード
    csv = display_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 CSVダウンロード",
        data=csv,
        file_name=f"類似駅_{target}.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown(f"#### 地図（{target}駅 と 類似駅TOP{top_n}）")

    # 地図データ作成
    map_data = []
    # 基準駅
    map_data.append({
        "駅名": f"★ {target}（基準）",
        "lat": target_data["lat"], "lon": target_data["lon"],
        "類似度": 1.0, "種別": "基準駅", "路線": target_data["路線"]
    })
    # 類似駅
    for _, row in result_df.iterrows():
        map_data.append({
            "駅名": row["駅名"],
            "lat": row["lat"], "lon": row["lon"],
            "類似度": row["類似度"], "種別": "類似駅", "路線": row["路線"]
        })
    map_df = pd.DataFrame(map_data)

    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon",
        hover_name="駅名",
        hover_data={"類似度": True, "路線": True, "lat": False, "lon": False},
        color="種別",
        color_discrete_map={"基準駅": "#e74c3c", "類似駅": "#2d6a4f"},
        size=[20 if t == "基準駅" else 12 for t in map_df["種別"]],
        zoom=10, height=500,
        mapbox_style="open-street-map"
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.markdown(f"#### 商圏クラスタ図（PCA 2次元）")
    st.caption(f"PC1寄与率: {pca_ratio[0]:.1%}　PC2寄与率: {pca_ratio[1]:.1%}")

    # 基準駅を強調
    df_plot = df_clustered.copy()
    df_plot["サイズ"] = df_plot["駅名"].apply(lambda x: 20 if x == target else 10)
    df_plot["強調"] = df_plot["駅名"].apply(lambda x: "★ 基準駅" if x == target else f"クラスタ {df_plot[df_plot['駅名']==x]['クラスタ'].values[0]}")

    fig_pca = px.scatter(
        df_plot, x="PCA_X", y="PCA_Y",
        color="クラスタ", text="駅名",
        size="サイズ",
        hover_data={"乗降客数": True, "地価": True, "PCA_X": False, "PCA_Y": False, "サイズ": False},
        height=500,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_pca.update_traces(textposition="top center", textfont_size=10)
    fig_pca.update_layout(
        xaxis_title="第1主成分（商圏規模）",
        yaxis_title="第2主成分（商圏特性）",
        showlegend=True
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# フッター
st.markdown("---")
st.caption("📌 データ出典：国交省国土数値情報・不動産情報ライブラリ・経済センサス（現在はサンプルデータ）")
