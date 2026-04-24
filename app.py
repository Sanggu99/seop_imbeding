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

def resize_image(image, max_size=1024):
    """AI 분석을 위해 이미지 크기를 최적화합니다."""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def perform_search(query_text):
    """공통 검색 및 결과 출력 로직"""
    if not query_text:
        st.warning("검색 내용이 없습니다.")
        return

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
                "match_threshold": 0.1,
                "match_count": num_results
            }
            
            rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/match_images"
            response = requests.post(rpc_url, headers=headers, json=payload, timeout=25)
            
            if response.status_code == 200:
                results = response.json()
                if results:
                    st.markdown(f"### 🔍 검색 결과 (총 {len(results)}건)")
                    st.markdown("---")
                    
                    images_b64 = [r.get("thumbnail_b64", "") for r in results]
                    titles = [r.get("project_name", "Unknown") for r in results]
                    
                    clicked_idx = clickable_images(
                        images_b64,
                        titles=titles,
                        div_style={"display": "grid", "grid-template-columns": f"repeat({grid_columns}, 1fr)", "gap": "15px"},
                        img_style={"width": "100%", "border-radius": "10px", "cursor": "zoom-in"},
                    )
                    
                    if clicked_idx > -1:
                        show_detail_popup(results[clicked_idx])
                else:
                    st.info("검색 결과가 없습니다.")
            else:
                st.error(f"서버 에러: {response.status_code}")
        except Exception as e:
            st.error(f"검색 중 오류 발생: {e}")

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

if search_mode == "📝 텍스트로 검색":
    query = st.text_input("검색어를 입력하세요:", placeholder="예: 화려한 도심 야경 속 커튼월 구조의 초고층 오피스 빌딩")
    if st.button("검색 시작") or (query and len(query) > 1):
        perform_search(query)

else:
    uploaded_file = st.file_uploader("참고할 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="업로드된 기준 이미지", width=300)
        if st.button("유사 이미지 검색"):
            with st.spinner("이미지 분석 중... (약 5~10초 소요)"):
                try:
                    raw_img = Image.open(uploaded_file)
                    optimized_img = resize_image(raw_img)
                    
                    vision_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    # 분위기와 색감을 최우선으로 분석하는 프롬프트
                    prompt = """
                    이 건축 이미지의 '분위기'와 '조명'을 중심으로 분석해줘.
                    
                    다음 순서로 상세히 설명해줘:
                    1. 지배적인 색상과 조명 상태 (예: 강렬한 주황색 노을, 황금빛 골든아워, 부드러운 저녁 빛)
                    2. 전체적인 무드와 시간대 (예: 몽환적인 석양 분위기, 평화로운 해질녘)
                    3. 건축물의 대략적인 특징과 재료 (예: 현대적인 대규모 건물, 유리와 금속)
                    
                    한국어로 답해줘.
                    """
                    response = vision_model.generate_content([prompt, optimized_img])
                    query_text = response.text
                    
                    # 무드 키워드 가중치 부여 로직 (검색어 맨 앞에 강력한 반복)
                    mood_prefix = ""
                    if any(word in query_text for word in ["노을", "석양", "해질녘", "골든아워"]):
                        mood_prefix = "Sunset Golden-hour Warm-lighting Orange-sky Sunset Golden-hour "
                    elif any(word in query_text for word in ["밤", "야경"]):
                        mood_prefix = "Night-view Dark-atmosphere Evening-lighting Night-view "
                    elif any(word in query_text for word in ["푸른", "낮", "맑은"]):
                        mood_prefix = "Clear-sky Bright-daylight Blue-sky "
                        
                    # 최종 쿼리: [무드 접두어 반복] + [상세 분석 텍스트]
                    enhanced_query = mood_prefix + query_text

                    st.success(f"🔍 **AI 분위기 분석 완료:**\n\n{query_text}")
                    
                    # 보강된 쿼리로 검색 실행
                    perform_search(enhanced_query)
                except Exception as e:
                    st.error(f"이미지 분석 실패: {e}")

# Footer
st.divider()
st.caption("© 2024 SEOP Architecture AI Archive. Powered by Gemini & Supabase.")
