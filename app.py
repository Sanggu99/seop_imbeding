import os
import streamlit as st
import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO
from st_clickable_images import clickable_images
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup Config
st.set_page_config(page_title="SEOP ARCHIVE", layout="wide")

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .stImage { border-radius: 12px; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
</style>
""", unsafe_allow_html=True)

# API Keys & Config (Prioritize st.secrets for Cloud)
def get_secret(key, default=""):
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key, default)

API_KEY = get_secret("GEMINI_API_KEY")
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

missing_keys = []
if not API_KEY: missing_keys.append("GEMINI_API_KEY")
if not SUPABASE_URL: missing_keys.append("SUPABASE_URL")
if not SUPABASE_KEY: missing_keys.append("SUPABASE_KEY")

if missing_keys:
    st.error(f"🔑 설정값이 누락되었습니다: {', '.join(missing_keys)}")
    st.info("""
    **해결 방법:**
    Streamlit Cloud의 **Settings > Secrets**에 아래 형식을 맞춰 입력해주세요:
    ```toml
    GEMINI_API_KEY = \"your_api_key\"
    SUPABASE_URL = \"your_supabase_url\"
    SUPABASE_KEY = \"your_supabase_key\"
    ```
    """)
    st.stop()

genai.configure(api_key=API_KEY)
EMBEDDING_MODEL = "models/gemini-embedding-001"

def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']

@st.dialog("📋 프로젝트 상세 정보", width="large")
def show_detail_popup(row):
    similarity = row.get('similarity', 0) * 100
    img_url = row.get('image_url', "")
    
    col_img, col_txt = st.columns([1.5, 1])
    with col_img:
        if img_url:
            st.image(img_url, use_container_width=True, caption=row.get('project_name'))
        else:
            st.error("이미지 주소를 찾을 수 없습니다.")
            
    with col_txt:
        st.markdown(f"### {row.get('project_name')}")
        st.markdown(f"**📍 용도:** {row.get('project_usage')}")
        st.markdown(f"**🎥 구도:** {row.get('camera_angle')}")
        st.markdown("---")
        st.markdown(f"**🏛️ 형태:** {row.get('massing_and_form')}")
        st.markdown(f"**🧱 마감재:** {row.get('materiality')}")
        st.markdown(f"**💡 분위기:** {row.get('lighting_and_atmosphere')}")
        st.markdown(f"**🌳 주변:** {row.get('surroundings')}")
        st.markdown(f"**✨ 키워드:** {row.get('style_keywords')}")
        
    st.markdown("#### 📝 VLM 심층 묘사 텍스트")
    st.info(row.get('embedding_text'))
    st.caption(f"클라우드 주소: {img_url} | 유사도: {similarity:.2f}%")

# Sidebar settings
st.sidebar.title("🔍 Search Options")
search_mode = st.sidebar.radio("검색 방식", ["📝 텍스트로 검색", "🖼️ 이미지로 검색"])

num_results = st.sidebar.slider("검색 결과 개수", min_value=12, max_value=300, value=60, step=12)
grid_columns = st.sidebar.select_slider("그리드 단수 (Columns)", options=[2, 3, 4, 5, 6], value=4)

st.title("SEOP ARCHIVE : Semantic Search 🏛️")
st.caption("AI 기반 클라우드 건축 이미지 아카이브 : 사진 한 장으로 유사한 디자인을 찾아보세요")

query_text = ""
if search_mode == "📝 텍스트로 검색":
    query = st.text_input("검색어를 입력하세요:", placeholder="예: 주변에 숲이 있고 따뜻한 무드의 주택 조감도")
    if query:
        query_text = query
else:
    uploaded_file = st.file_uploader("참고할 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="업로드된 기준 이미지", width=300)
        if st.button("유사 이미지 검색"):
            with st.spinner("이미지 분석 중..."):
                try:
                    img = Image.open(uploaded_file)
                    try:
                        vision_model = genai.GenerativeModel('gemini-1.5-flash')
                        response = vision_model.generate_content([
                            "Describe this architecture image in detail, focusing on materials, lighting, style, and atmosphere for semantic search.",
                            img
                        ])
                    except Exception:
                        # Fallback for older API versions
                        vision_model = genai.GenerativeModel('gemini-pro-vision')
                        response = vision_model.generate_content([
                            "Describe this architecture image in detail, focusing on materials, lighting, style, and atmosphere for semantic search.",
                            img
                        ])
                    query_text = response.text
                    st.info(f"🔍 **이미지 분석:** {query_text[:100]}...")
                except Exception as e:
                    st.error(f"이미지 분석 실패: {e}")

if query_text:
    with st.spinner("클라우드 벡터 공간 검색 중..."):
        try:
            # 1. Get query embedding
            query_emb = get_embedding(query_text)
            
            # 2. Call Supabase RPC function
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "query_embedding": query_emb,
                "match_threshold": 0.1,  # 최소 유사도
                "match_count": num_results
            }
            
            rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/match_images"
            response = requests.post(rpc_url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                results = response.json()
                
                if results:
                    st.markdown(f"### 🔍 '{query}' 검색 결과 (총 {len(results)}건)")
                    st.markdown("---")
                    
                    # Display Grid
                    images_b64 = [r.get("thumbnail_b64", "") for r in results]
                    titles = [r.get("project_name", "Unknown") for r in results]
                    
                    clicked_idx = clickable_images(
                        images_b64,
                        titles=titles,
                        div_style={"display": "grid", "grid-template-columns": f"repeat({grid_columns}, 1fr)", "gap": "15px"},
                        img_style={"width": "100%", "border-radius": "10px", "cursor": "zoom-in", "transition": "transform 0.2s"},
                    )
                    
                    if clicked_idx > -1:
                        show_detail_popup(results[clicked_idx])
                else:
                    st.warning("일치하는 결과가 없습니다.")
            else:
                st.error(f"Supabase Error ({response.status_code}): {response.text}")
                
        except Exception as e:
            st.error(f"Error: {e}")
