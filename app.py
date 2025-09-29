import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import requests
import json
import openai
from google import genai

from streamlit_plotly_events import plotly_events

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
# 주의: API 키를 코드에 직접 하드코딩하는 것은 보안상 위험합니다.
# 실제 운영 환경에서는 st.secrets 또는 환경 변수를 사용하세요.
GEMINI_API_KEY = st.secrets.get("GEMINI_ApI_KEY") 
SERPER_API_KEY = st.secrets.get("SERPER_ApI_KEY")
FILE_NAME = "키워드_최종_종합트렌드_강화로직.csv"

CLASSIFICATION_CRITERIA = ["접근방식 기준", "건축설계단계", "공간적 스케일"]
COLOR_BASE_COLUMN = "접근방식 기준"
SIZE_COLUMN = "Final Trend Index" # 버블 크기 기준 컬럼
ACADEMIC_COL = "Academic_Total"
MEDIA_COL = "Media_TotalY"

FIXED_COLOR_MAP = {
    "사용자": "#FFD700",
    "스마트 기술": "#1E90FF",
    "공간 구성": "#FF8C00",
    "지속가능성": "#3CB371",
    "건축 기술": "#808080",
    "브랜드 & 서비스": "#9932CC",
    "모듈러": "#CCCCCC"
}

# ----------------------------------------------------------------------
# 데이터
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(FILE_NAME, encoding="utf-8-sig")
    for c in [SIZE_COLUMN, ACADEMIC_COL, MEDIA_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in CLASSIFICATION_CRITERIA:
        if c not in df.columns:
            df[c] = "N/A"
        else:
            df[c] = df[c].fillna("N/A")
    if "Keyword" not in df.columns:
        raise ValueError("CSV에 'Keyword' 칼럼이 없습니다.")
    return df

df = load_data()

# ----------------------------------------------------------------------
# LLM (Gemini) 정의 생성
# ----------------------------------------------------------------------
def generate_definition_with_llm(keyword, _log_cb=lambda x: None):
    """
    1. 정의: Gemini API로 생성 (실제 Google GenAI API 호출).
    """
    if not GEMINI_API_KEY: # ✨ 수정
        msg = "[오류] GEMINI_API_KEY가 설정되지 않았거나 유효하지 않습니다." # ✨ 수정
        return msg

    try:
        client = genai.Client(api_key=GEMINI_API_KEY) # ✨ 수정: Gemini 클라이언트 초기화

        system_prompt = (
            "당신은 건축, 도시, 공간 디자인 분야의 최신 트렌드 전문가입니다. "
            "사용자가 제시한 키워드에 대해 명확하고 간결하며 전문적인 정의를 한국어로 3~5문장 이내로 작성하세요."
        )
        user_prompt = f"키워드 '{keyword}'에 대한 정의를 작성해 주세요."

        # response = client.chat.completions.create(...) # 기존 코드 (복잡)
        response = client.models.generate_content( # ✨ 수정: generate_content 사용
            model="gemini-2.5-flash", # ✨ 수정: Gemini 모델로 변경
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt, # ✨ 수정: system_instruction 사용
                temperature=0.3,
                max_output_tokens=300 # max_tokens 대신 max_output_tokens 사용
            )
        )
        
        # result = response.choices[0].message.content.strip() # 기존 코드
        result = response.text.strip() # ✨ 수정: response.text 사용

        if not result:
            raise ValueError("LLM이 정의를 생성하지 못했습니다.")
            
        return result
        
    # except openai.AuthenticationError: # 기존 코드
    except genai.errors.APIError as e: # ✨ 수정: Gemini API 오류 처리
        msg = f"[Gemini API 오류] 키가 유효하지 않거나 요청에 실패했습니다. 키를 확인하세요: {e}"
        return msg
    except Exception as e:
        msg = f"[LLM 처리 오류: {e}]"
        return "LLM 정의 생성에 실패했습니다: " + str(e)
# ----------------------------------------------------------------------
# Serper
# ----------------------------------------------------------------------
def get_serper_info(keyword, search_type, _log_cb=lambda x: None):
    """
    2. 뉴스: Serper API 활용 (최신/인기 검색 강화)
    3. 논문: Serper API 활용 (학술 검색 강화)
    """
    url = "https://google.serper.dev/search"
    query, num = "", 5
    tbm_type = None

    if not SERPER_API_KEY:
        msg = "[오류] SERPER_API_KEY가 설정되지 않았거나 유효하지 않습니다."
        return msg
    
    if search_type == "definition":
        return "Serper를 통한 정의 검색은 LLM으로 대체되었습니다."
    elif search_type == "news":
        query = f"{keyword} 최신 트렌드"
        tbm_type = "news" 
        num = 10 
    elif search_type == "scholar":
        query = f"{keyword} 학술 논문"
        num = 10 
    else:
        return "지원하지 않는 search_type"

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "hl": "ko", "num": num}
    if tbm_type:
        payload["tbm"] = tbm_type

    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        data = r.json()

        key = "news" if search_type == "news" else "organic"
        items = data.get(key, []) or []
        
        if search_type == "scholar":
            def is_scholar(item):
                title = item.get("title", "").lower()
                link = item.get("link", "").lower()
                return (
                    "논문" in title or "연구" in title or "저널" in title or 
                    "scholar.google" in link or "doi.org" in link or "researchgate" in link
                )
            items = [it for it in items if is_scholar(it)]
            
        results = [{"title": it.get("title", "(제목 없음)"), "link": it.get("link", "#")} for it in items[:5]]
        
        if not results and search_type in ["news", "scholar"]:
            payload.pop("tbm", None) 
            payload["num"] = 5
            
            if search_type == "scholar":
                payload["q"] = f"{keyword} 학술 논문" 
            else: 
                payload["q"] = f"{keyword} 최신 트렌드"
                
            r_fallback = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            r_fallback.raise_for_status()
            data_fallback = r_fallback.json()
            items_fallback = data_fallback.get("organic", [])
            
            if search_type == "scholar":
                 items_fallback = [it for it in items_fallback if is_scholar(it)]
                 
            results = [{"title": it.get("title", "(제목 없음)"), "link": it.get("link", "#")} for it in items_fallback[:5]]

        return results

    except requests.exceptions.RequestException as e:
        status = getattr(e.response, "status_code", "N/A")
        msg = f"[Serper API 오류: status={status}, {e}]"
        return msg
    except Exception as e:
        msg = f"[처리 오류: {e}]"
        return msg

# ----------------------------------------------------------------------
# 그래프 (히트박스 + 시각 레이어) - 애니메이션 최적화 (3개 Trace)
# ----------------------------------------------------------------------
def create_keyword_figure(df, classification_criteria, size_criteria, color_base_column):
    G = nx.Graph()

    for _, row in df.iterrows():
        kw = str(row["Keyword"])
        G.add_node(
            kw,
            size=float(row[size_criteria]),
            group=str(row[classification_criteria]),
            color_group=str(row[color_base_column]),
        )

    for group_name, group_df in df.groupby(classification_criteria):
        group_keywords = group_df["Keyword"].astype(str).tolist()
        if len(group_keywords) > 1:
            leader_row = group_df.loc[group_df[size_criteria].idxmax()]
            leader = str(leader_row["Keyword"])
            for kw in group_keywords:
                if kw != leader:
                    G.add_edge(leader, kw)

    # k=0.55 -> k=0.8로 증가시켜 노드 간격을 넓힘
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=60)
    
    all_nodes_sorted = sorted(G.nodes())

    # 1. 엣지 Trace
    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(150,150,160,0.25)"),
        hoverinfo="skip",
        showlegend=False
    )
    traces = [edge_trace]

    # 2. 노드 데이터 통합
    all_xs, all_ys, all_sizes, all_texts, all_colors, all_customdata, all_hovertext = [], [], [], [], [], [], []
    node_sizes_all = [G.nodes[n]["size"] for n in G.nodes()]
    sizeref = 2.0 * max(node_sizes_all) / (80.0 ** 2) if node_sizes_all else 1

    for kw in all_nodes_sorted:
        node_data = G.nodes[kw]
        
        all_xs.append(pos[kw][0])
        all_ys.append(pos[kw][1])
        all_sizes.append(max(10.0, node_data["size"]))
        all_texts.append(kw)
        all_colors.append(FIXED_COLOR_MAP.get(node_data["color_group"], "#CCCCCC"))
        all_customdata.append(kw) 
        all_hovertext.append(f"키워드: {kw}<br>그룹: {node_data['group']}<br>크기: {int(node_data['size'])}")

    # 3. 마커 레이어 (버블)
    marker_trace = go.Scatter(
        x=all_xs, y=all_ys,
        mode="markers",
        hoverinfo="text",
        hovertext=all_hovertext,
        marker=dict(
            size=all_sizes,
            sizemode="area",
            sizeref=sizeref,
            sizemin=10,
            color=all_colors,
            line=dict(width=1, color="rgba(20,20,20,0.6)"),
            opacity=0.95,
        ),
        showlegend=False, 
        name="Markers",
        customdata=all_customdata, 
    )
    traces.append(marker_trace)
    
    # 4. 텍스트 레이어 (키워드 이름)
    text_trace = go.Scatter(
        x=all_xs, y=all_ys,
        mode="text",
        text=all_texts,
        textposition="middle center",
        textfont=dict(size=10, color="black"),
        hoverinfo="skip", 
        showlegend=False,
        name="Text",
        customdata=all_customdata, 
    )
    traces.append(text_trace)

    # 5. 히트박스 레이어 (클릭 영역)
    all_hit_sizes = [s * 1.6 + 20 for s in all_sizes]
    hit_trace = go.Scatter(
        x=all_xs, y=all_ys,
        mode="markers",
        marker=dict(
            size=all_hit_sizes,
            sizemode="area",
            sizeref=sizeref,
            sizemin=20,
            color="rgba(0,0,0,0.01)", 
            line=dict(width=0),
        ),
        hoverinfo="skip",
        showlegend=False,
        name="Hitbox",
        customdata=all_customdata 
    )
    traces.append(hit_trace)
    
    # 6. Figure Layout 설정 (Transition 포함)
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text=f"키워드 네트워크 (그룹: {classification_criteria}, 색상: {color_base_column})",
                font=dict(size=18),
            ),
            margin=dict(b=20, l=5, t=50, r=150), 
            
            annotations=[
                dict(
                    xref='paper', yref='paper',
                    x=1.02, y=0.98 - i*0.04,
                    text=f'<span style="color:{FIXED_COLOR_MAP[group]}; font-size:12px;">\u25CF</span> {group}',
                    showarrow=False,
                    align="left",
                    visible=group in df[color_base_column].unique() 
                ) for i, group in enumerate(FIXED_COLOR_MAP.keys())
            ],
            showlegend=False, 
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            clickmode="event+select",
            dragmode="pan",
            transition=dict(
                duration=700, 
                easing="cubic-in-out" 
            )
        ),
    )
    return fig

# ----------------------------------------------------------------------
# 이벤트에서 키워드 복원 (가장 견고하게)
# (변동 없음)
# ----------------------------------------------------------------------
def extract_keyword_from_events(events, fig: go.Figure):
    if not events or not isinstance(events, list):
        return None, None
    ev = events[0] or {}
    # 1) customdata 직접
    cd = ev.get("customdata", None)
    if isinstance(cd, list):
        if cd:
            return cd[0], ev
    elif isinstance(cd, str):
        return cd, ev
    # 2) curveNumber/pointNumber로 fig.data에서 역참조
    curve = ev.get("curveNumber", None)
    pnum = ev.get("pointNumber", ev.get("pointIndex", None))
    try:
        if curve is not None and pnum is not None:
            trace = fig.data[int(curve)]
            # 우선 customdata → 없으면 text
            if hasattr(trace, "customdata") and trace.customdata is not None:
                val = trace.customdata[int(pnum)]
                if isinstance(val, (str, int, float)):
                    return str(val), ev
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    return str(val[0]), ev
            if hasattr(trace, "text") and trace.text is not None:
                txt = trace.text[int(pnum)]
                if isinstance(txt, (str, int, float)):
                    return str(txt), ev
    except Exception:
        pass
    # 3) 최후: trace 이름(부정확 가능)
    if curve is not None:
        try:
            name = fig.data[int(curve)].name
            return str(name), ev
        except Exception:
            pass
    return None, ev

# ----------------------------------------------------------------------
# 앱
# ----------------------------------------------------------------------
def run():
    st.set_page_config(layout="wide", page_title="키워드 트렌드 시각화")
    st.title("📊 키워드 분석 및 시각화")
    st.caption("그래프에서 **키워드**를 클릭하면 아래에 정의/뉴스/논문이 표시됩니다.")

    # 키워드/API 설정 가이드 (하드코딩된 키의 경우 경고는 유지)
    if not GEMINI_API_KEY or not SERPER_API_KEY:
        st.error("⚠️ **API 키 설정 필요:** GEMINI_API_KEY 또는 SERPER_API_KEY가 설정되지 않았습니다.")
    
    with st.sidebar:
        st.header("🔍 분류 기준")
        crit = st.radio("그룹핑 기준", CLASSIFICATION_CRITERIA, index=0)

        # ====== 추가된 팝업 버튼 로직 ======
        st.markdown("---")
        with st.popover("키워드 버블 크기 기준"):
            st.markdown(f"**버블 크기 기준:**")
            st.markdown(f"키워드 버블의 크기는 **'{SIZE_COLUMN}'** 값에 비례하여 설정됩니다.")
            st.markdown(f"이는 학술 연구 및 미디어 노출도를 종합한 **최종 트렌드 지수**를 나타냅니다.")
            st.markdown(f"**A. 트렌드 점수** = 대중의 관심도 및 미래 성장세 평가. Google Trends를 사용하며, 최근 검색량이 이전보다 얼마나 폭발적으로 늘었고(성장률), 그 상승세가 얼마나 가파르게 유지되는지(모멘텀)를 봅니다. / 성장률 (70%) + 모멘텀 (30%)")
            st.markdown(f"**B. 미디어 점수** = 대중 매체를 통한 확산 정도 평가. 뉴스/기사 건수를 사용하며, 최근 1개월간의 기사 증가세(성장률)와 전체 1년간의 기사 규모(총량)를 봅니다. / 성장률 (60%) + 총량 (40%)")
            st.markdown(f"**C. 학술 점수** = 최근 3년간의 논문 건수를 사용하며, 해당 기간의 연구 증가 속도(성장률)와 전체 연구량(총량)을 봅니다. / 성장률 (60%) + 총량 (40%)")
            st.markdown(f"세 점수가 산정되면, 아래의 공식에 따라 최종 트렌드 지수를 계산합니다.")
            st.markdown(f"**최종 트렌드 지수=(0.5×트렌드 점수)+(0.25×미디어 점수)+(0.25×학술 점수)**")


        # ==================================

    fig = create_keyword_figure(df.copy(), crit, SIZE_COLUMN, COLOR_BASE_COLUMN)

    events = plotly_events(
        fig,
        click_event=True,
        select_event=True,
        hover_event=False,
        override_height=720,
        override_width=1100,
        key="interactive_graph_robust",
    )

    # (디버그) 원시 이벤트
    #with st.expander("🛠 이벤트 원시 로그 (디버그)"):
        #st.write(events)

    # 키워드 복원
    clicked_keyword, ev_used = extract_keyword_from_events(events, fig)

    result_area = st.container()
    if clicked_keyword:
        with result_area:
            st.markdown("---")
            st.subheader(f"🔎 선택 키워드: **{clicked_keyword}**")
            
            null_log = lambda x: None 
            
            with st.spinner("검색 및 정의 생성 중입니다…"):
                definition = generate_definition_with_llm(clicked_keyword, _log_cb=null_log)
                news = get_serper_info(clicked_keyword, "news", _log_cb=null_log)
                scholar = get_serper_info(clicked_keyword, "scholar", _log_cb=null_log)

            is_error = "[오류]" in definition or "[Serper API 오류]" in str(news) or "[Serper API 오류]" in str(scholar)
            if is_error:
                 st.error("일부 기능에서 오류가 발생했습니다. 키 설정 및 API 응답을 확인하세요.")
            else:
                 st.success("검색 및 정의 생성 완료")

            t1, t2, t3 = st.tabs(["정의 (Chat GPT)", "최신/인기 뉴스", "학술 논문"])
            with t1:
                st.markdown(f"**Chat GPT 생성 정의**\n\n> {definition}")
            with t2:
                if isinstance(news, list) and news and not is_error:
                    for it in news:
                        st.markdown(f"- [{it['title']}]({it['link']})")
                else:
                    st.warning("뉴스 검색 결과를 찾지 못했습니다. 키워드나 API 응답을 확인하세요.")
            with t3:
                if isinstance(scholar, list) and scholar and not is_error:
                    for it in scholar:
                        st.markdown(f"- [{it['title']}]({it['link']})")
                else:
                    st.warning("논문 검색 결과를 찾지 못했습니다. 키워드나 API 응답을 확인하세요.")
    else:
        with result_area:
            st.info("그래프에서 키워드를 클릭해 주세요.")

if __name__ == "__main__":

    run()

