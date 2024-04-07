import streamlit as st
import pandas as pd
#from gensim import corpora, models
#import pyLDAvis
#import pyLDAvis.gensim_models as gensimvis
#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

# nltk의 stopwords를 다운로드합니다.
#nltk.download('stopwords')
#nltk.download('punkt')

# 페이지 설정
st.set_page_config(page_title="분석 시스템", layout="wide")

# 전체 페이지에 대한 CSS 스타일 적용
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF !important;
    }
    .stSidebar > div:first-child {
        background-color: #007bff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 사용자가 원하는 순서대로 사이드바 메뉴 구성
menu = st.sidebar.selectbox('메뉴를 선택하세요', ['홈', '사용방법', '데이터 수집', '클러스터링', '사회 연결망 분석', '토픽 모델링', '기회점수'])

# 전역 변수로 데이터프레임 초기화
df = None

if menu == '홈':
    #st.title('애플리케이션 홈')
    col1, col2 = st.columns(2)
    with col1:
        st.image('https://brassy-dolphin-f22.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fa60d437f-ec9f-4f91-a164-6f05ff0d8852%2F2ea475da-6a76-45ea-9471-e6b42a36edf8%2FUntitled.png?table=block&id=fa04a16a-d655-43aa-93c1-cc6c1c90ff3f&spaceId=a60d437f-ec9f-4f91-a164-6f05ff0d8852&width=500&userId=&cache=v2', width=300)
        st.image('https://building2.auric.or.kr/Upload/BuildingImg/Large/6601_92761490-a595-49fd-81ed-a8f6a1d4c413.jpg?width=900', width=800)
    
elif menu == '사용방법':
    st.title('사용방법')
    st.write('DGB 고객 분석 시스템 사용방법 입니다.')
    
    st.write('1.수집한 데이터를 업로드 해주세요')
    st.write('2.수집한 데이터를 업로드 해주세요')

elif menu == '데이터 수집':
    st.title('데이터 수집')
    uploaded_file = st.file_uploader("파일을 업로드하세요.", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.write("업로드된 파일의 데이터프레임:")
            st.dataframe(df)
            
elif menu == '토픽 모델링':
    st.title('토픽 모델링')
    if df is not None:
        # 데이터 전처리
        # 텍스트 데이터가 포함된 열의 이름을 'text'으로 가정합니다.
        data = df['text'].values.tolist()
        stop_words = set(stopwords.words('english'))
        texts = [[word for word in word_tokenize(doc.lower()) if word not in stop_words and word.isalpha()] for doc in data]
        
        # LDA 모델 훈련
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        
        # PyLDAvis를 사용한 시각화
        vis = gensimvis.prepare(lda_model, corpus, dictionary)
        pyldavis_html = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(pyldavis_html, width=1300, height=800, scrolling=True)
    else:
        st.write("먼저 '데이터 수집' 메뉴에서 데이터를 업로드해주세요.")

