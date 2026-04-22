import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Embedding", layout="wide")

# Secrets handling for Cloud
def get_secret(key, default=""):
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

@st.cache_resource
def load_db():
    missing = []
    if not SUPABASE_URL: missing.append("SUPABASE_URL")
    if not SUPABASE_KEY: missing.append("SUPABASE_KEY")
    
    if missing:
        st.warning(f"⚠️ 설정값이 누락되었습니다: {', '.join(missing)}")
        return None
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    try:
        response = requests.get(f"{SUPABASE_URL}/rest/v1/architecture_images?select=id,project_name,project_usage,materiality,lighting_and_atmosphere,style_keywords,embedding", headers=headers, timeout=20)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"📡 데이터 로드 실패 (Status: {response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ 연결 오류: {e}")
        return None

data = load_db()

st.title("🌌 Architecture Embedding")
st.markdown("수백 개의 프로젝트 이미지가 내포하고 있는 의미(Semantic)를 인공지능이 어떻게 군집화(Clustering)하고 있는지 탐색해보세요.")

if data is None:
    st.error("데이터를 불러올 수 없습니다. 설정 또는 네트워크 상태를 확인해주세요.")
    st.stop()

with st.spinner("DB에서 모든 벡터 데이터를 불러오고 있습니다..."):
    embeddings = []
    metadatas = []
    ids = []
    for row in data:
        # Check string representation or list
        emb = row.get('embedding')
        if isinstance(emb, str):
            emb = json.loads(emb)
        embeddings.append(emb)
        metadatas.append(row)
        ids.append(row.get('id'))

if not embeddings or len(embeddings) == 0:
    st.warning("데이터베이스에 임베딩된 데이터가 없습니다.")
    st.stop()

# --- Data Cleaning & Grouping Logic ---
def simplify_usage(usage):
    if not usage: return "기타"
    u = usage.lower()
    if any(x in u for x in ["문화", "집회", "전시", "도서관", "복지", "종교"]): return "문화/집회/복지"
    if any(x in u for x in ["업무", "오피스", "청사", "사옥"]): return "업무/공공청사"
    if any(x in u for x in ["상업", "판매", "근린", "리테일", "시장", "휴게"]): return "상업/판매/휴게"
    if any(x in u for x in ["주거", "주택", "아파트", "숙박", "보금자리"]): return "주거/숙박"
    if any(x in u for x in ["교육", "연구", "학교", "캠퍼스"]): return "교육/연구"
    if any(x in u for x in ["의료", "치유", "병원", "보건"]): return "의료/보건"
    if any(x in u for x in ["주차", "교통", "역"]): return "교통/주차"
    if any(x in u for x in ["체육", "스포츠", "관광", "레저"]): return "체육/관광/레저"
    return "기타 용도"

def simplify_material(m):
    if not m or m == "None": return "기타 마감재"
    m = m.split(",")[0].strip().lower()
    if any(x in m for x in ["콘크리트", "노출", "concrete"]): return "노출콘크리트"
    if any(x in m for x in ["벽돌", "브릭", "조적", "brick"]): return "벽돌/조적"
    if any(x in m for x in ["나무", "우드", "루버", "목재", "wood"]): return "우드/목재"
    if any(x in m for x in ["유리", "커튼월", "글래스", "glass"]): return "유리/커튼월"
    if any(x in m for x in ["금속", "메탈", "알루미늄", "패널", "징크", "강판", "타공"]): return "금속/외장패널"
    if any(x in m for x in ["돌", "석재", "대리석", "라임스톤", "stone"]): return "석재/돌"
    if any(x in m for x in ["백색", "화이트", "도장", "페인트", "테라코타"]): return "밝은 마감/도장"
    return "기타 마감재"

def simplify_mood(m):
    if not m or m == "None": return "일반 무드"
    m = m.split(",")[0].strip().lower()
    if any(x in m for x in ["자연", "햇빛", "한낮", "밝은", "자연광", "맑은", "주간", "화창한", "청명한"]): return "밝은 자연광/한낮"
    if any(x in m for x in ["저녁", "매직아워", "노을", "석양", "따뜻한", "골든아워", "해질녘"]): return "매직아워/노을빛"
    if any(x in m for x in ["밤", "야경", "어두운", "조명", "야간", "은은한"]): return "야간/인공조명"
    if any(x in m for x in ["안개", "차분한", "흐린", "비", "눈", "새벽"]): return "차분함/안개/흐림"
    if any(x in m for x in ["대비", "강렬한", "극적인", "드라마틱"]): return "극적인 명암 대비"
    return "일반 무드"

def simplify_concept(m):
    if not m or m == "None": return "기타 컨셉"
    m = m.split(",")[0].strip().lower()
    if any(x in m for x in ["모던", "현대", "심플", "세련된", "미니멀"]): return "모던/미니멀"
    if any(x in m for x in ["자연", "친환경", "유기적인", "편안한", "생태", "부드러운"]): return "자연친화/유기적"
    if any(x in m for x in ["전통", "클래식", "고전", "역사", "감성적인"]): return "전통/감성적"
    if any(x in m for x in ["미래", "하이테크", "첨단", "혁신", "독창적", "곡선형"]): return "미래지향/독창적"
    if any(x in m for x in ["역동", "유동", "활동적인", "생동감", "흐름", "활발한", "활기찬"]): return "역동적/생동감"
    if any(x in m for x in ["개방", "투명", "열린", "시원한", "수평적인"]): return "개방적/수평적"
    if any(x in m for x in ["상징", "기념비", "웅장", "단단한", "견고한"]): return "상징적/웅장함"
    return "기타 컨셉"

def simplify_project_name(name):
    if not name: return "Unknown"
    words_to_remove = [" 건립사업", " 신축공사", " 설계공모", " 건축설계", " 제안공모", " 국제설계공모", " 마스터플랜", " 설계용역", "조성사업"]
    new_name = name
    for w in words_to_remove:
        new_name = new_name.replace(w, "")
    return new_name.strip()

# --- Sidebar UI ---
st.sidebar.title("🛠️ 시각화 설정")
dim_red_method = st.sidebar.radio("차원 축소 알고리즘", ["t-SNE (추천, 국소적 군집 파악)", "PCA (빠름, 전역적 분포)"])
n_components = st.sidebar.radio("차원 (Dimensions)", ["3D 공간", "2D 공간"], index=0)
is_3d = n_components == "3D 공간"
dim_val = 3 if is_3d else 2

st.sidebar.markdown("---")
color_by_options = {
    "용도 (Usage Category)": 'Usage Category',
    "문맥/분위기 (Atmosphere)": 'Primary Mood',
    "핵심 재질 (Material)": 'Primary Material',
    "디자인 컨셉 (Concept)": 'Primary Concept',
    "단일 색상 (None)": None
}
color_by_label = st.sidebar.selectbox("색상 그룹핑 기준 (Clustering)", list(color_by_options.keys()), index=0)
color_col = color_by_options[color_by_label]

@st.cache_data
def reduce_dimensions(embs, method, dim):
    X = np.array(embs)
    if method.startswith("t-SNE"):
        perplexity = min(30, max(2, len(X) - 1))
        model = TSNE(n_components=dim, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    else:
        model = PCA(n_components=dim, random_state=42)
    return model.fit_transform(X)

with st.spinner(f"768차원의 인공지능 벡터를 {n_components}로 축소 압축 중입니다..."):
    X_reduced = reduce_dimensions(embeddings, dim_red_method, dim_val)

# Prepare DataFrame
df = pd.DataFrame()
df['x'] = X_reduced[:, 0]
df['y'] = X_reduced[:, 1]
if is_3d:
    df['z'] = X_reduced[:, 2]
    
df['id'] = ids
df['Project Name'] = [simplify_project_name(m.get('project_name', 'Unknown')) for m in metadatas]
df['Usage Category'] = [simplify_usage(m.get('project_usage', 'Unknown')) for m in metadatas]
df['Primary Material'] = [simplify_material(m.get('materiality', 'None')) for m in metadatas]
df['Primary Mood'] = [simplify_mood(m.get('lighting_and_atmosphere', 'None')) for m in metadatas]
df['Primary Concept'] = [simplify_concept(m.get('style_keywords', 'None')) for m in metadatas]

# --- Append Counts to Legend Labels ---
for col in ['Usage Category', 'Primary Material', 'Primary Mood', 'Primary Concept']:
    counts = df[col].value_counts()
    df[col] = df[col].apply(lambda x: f"{x} ({counts[x]})")

hover_data = {
    'x': False, 'y': False, 'id': False,
    'Project Name': True,
    'Usage Category': True,
    'Primary Material': True,
    'Primary Mood': True,
    'Primary Concept': True
}
if is_3d:
    hover_data['z'] = False

# --- Selection Logic to highlight entire color group ---
category_to_highlight = None

# We use Streamlit Session State to listen to Node Click
if "plotly_cluster_map" in st.session_state:
    evt = st.session_state["plotly_cluster_map"]
    if evt and evt.get("selection", {}).get("points", []):
        pt = evt["selection"]["points"][0]
        # Find which Category this clicked node belongs to by matching X and Y
        try:
            matched_row = df[(np.isclose(df['x'], pt['x'])) & (np.isclose(df['y'], pt['y']))]
            if not matched_row.empty and color_col:
                category_to_highlight = matched_row.iloc[0][color_col]
        except Exception as e:
            pass

# Draw Plot
plot_func = px.scatter_3d if is_3d else px.scatter
kwargs = dict(
    x='x', y='y', color=color_col,
    hover_name='Project Name', hover_data=hover_data,
    color_discrete_sequence=px.colors.qualitative.Pastel
)
if is_3d:
    kwargs['z'] = 'z'

fig = plot_func(df, **kwargs)

# Layout modifications including UIREVISION to prevent 3D camera reset!
if is_3d:
    fig.update_layout(
        height=750, margin=dict(l=0, r=0, b=0, t=0), 
        scene=dict(xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3'),
        uirevision="constant_camera",
        clickmode='event',
        legend=dict(yanchor="top", y=0.85, xanchor="left", x=1.02)
    )
else:
    fig.update_layout(
        height=750, margin=dict(l=0, r=0, b=0, t=0), 
        xaxis_title='Dim 1', yaxis_title='Dim 2',
        uirevision="constant_camera",
        clickmode='event',
        legend=dict(yanchor="top", y=0.85, xanchor="left", x=1.02)
    )

# Apply Opacity filtering natively to traces based on clicked category
if color_col and category_to_highlight:
    for trace in fig.data:
        if trace.name == str(category_to_highlight):
            trace.marker.opacity = 0.95
            trace.marker.line.width = 1.5
            trace.marker.line.color = 'DarkSlateGrey'
        else:
            trace.marker.opacity = 0.05
else:
    fig.update_traces(marker=dict(opacity=0.8, line=dict(color='DarkSlateGrey', width=1)))
    
if is_3d:
    fig.update_traces(marker=dict(size=6))
else:
    fig.update_traces(marker=dict(size=9))

# Render with on_select
if is_3d:
    st.plotly_chart(fig, use_container_width=True, key="plotly_cluster_map_3d")
    st.warning("⚠️ **3D 맵 한계 안내**: 3D 공간에서는 '점(노드)을 클릭해 선택'하는 기능이 원천적으로 지원되지 않습니다. 3D에서 클러스터 강조를 원하시면, 반드시 우측 **색상표(Legend) 글씨를 더블클릭** 하셔야 합니다.")
else:
    st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="plotly_cluster_map")
    st.info("💡 **2D 클러스터 강조 팁**: 노드(점)를 한 번 **클릭**하시면, 접속된 같은 색상의 클러스터 전체 군집만 하이라이트됩니다! 바탕을 클릭하면 원래대로 돌아옵니다.")

# --- Educational Explanation Section ---
with st.expander("📚 원리 및 활용 가이드 (t-SNE vs PCA, 축소 차원의 의미)", expanded=False):
    st.markdown("""
    ### 1. 차원 축소 알고리즘의 차이 (t-SNE vs PCA)
    * **PCA (주성분 분석)**: 데이터의 전반적인(Global) 수학적 뼈대나 윤곽을 그대로 유지하면서 축소합니다. 렌더링 속도가 매우 빠르지만, 복잡한 비선형 데이터에서는 군집(Cluster)이 겹쳐 보일 수 있습니다. (전체 데이터가 어떻게 퍼져있는지 볼 때 유리)
    * **t-SNE**: 데이터 간의 지역적인(Local) 이웃 관계에 집중하여 축소합니다. '원래 가까웠던 점은 끝까지 가깝게' 유지하려는 성질이 있어, 군집(Cluster)을 시각적으로 예쁘고 뚜렷하게 나누어 보여주는데 특화되어 있습니다. (성향이 비슷한 이미지들의 무리를 찾을 때 강력히 추천)

    ### 2. 3D/2D 공간에서 'Dim 1, 2, 3'은 무엇을 의미하나요?
    * AI 모델은 문장을 이해할 때 768차원(결과값 768개의 숫자)이라는 인간이 상상할 수 없는 고차원 축을 사용합니다.
    * 화면에 표시된 **Dim 1, Dim 2, Dim 3 (X,Y,Z 축)은 물리적인 길이, 높이를 의미하는 것이 아닙니다.** 수많은 차원을 **의미론적인 특성이 가장 크게 차이나는 상위 3가지 임의의 시각적 기준선**으로 압축해낸 추상적 좌표입니다. (예: 어떤 축은 '현대적vs클래식' 차이일 수 있고, 다른 축은 '조명톤'의 차이일 수도 있습니다.)

    ### 3. 점들이 가까이 뭉쳐있다는 건 어떤 의미인가요?
    * 3D/2D 맵에서 **거리가 가깝다는 것은 'VLM 이미지가 묘사한 느낌과 서사가 극도로 비슷하다는 것'**을 뜻합니다. 
    * 만약 A 프로젝트 렌더링 이미지와 B 프로젝트 이미지가 서로 다른 폴더에 있더라도, VLM이 보기에 "석양빛을 받는 곡선형 노출콘크리트 건물"로 비슷하게 묘사했다면 우주 공간의 바로 옆에 붙어서 렌더링됩니다.

    ### 4. 건축 설계 실무에서의 활용 방안 🚀
    1. **무의식적 디자인 패턴 도출 (Style Auditing)**:
        * 우리 사무소가 과거에 진행한 프로젝트들을 뿌려놓았을 때, **유독 한쪽에 거대하게 많은 군집**이 존재한다면, 그것이 곧 "SEOP 건축사가 무의식적으로 고수하고 있는 특유의 회사 디자인 언어(Tone & Manner)"입니다.
    2. **레퍼런스 이미지 자동 큐레이션 (Reference Retrieval)**:
        * 새로운 프로젝트 기획 단계에서 "이번엔 우드 데크를 쓴 친환경 느낌으로 가보자"라고 했을 때, 맵 상에서 '우드/친환경' 클러스터 영역을 줌인하면, 과거 그와 같은 컨셉으로 성공했던 프로젝트 산출물들이 모두 그곳에 모여있습니다.
    3. **데이터 편향 및 공백 확인**:
        * 점이 하나도 없는 빈 공간(Void)이 존재한다면, 해당 스타일이나 형태, 재질의 조합은 우리 회사가 아직 한 번도 시도해본 적 없는 디자인 영역임을 시각적으로 증명합니다. 새로운 공모전의 타겟팅 방향성을 세울 때 사용할 수 있습니다.
    """)
