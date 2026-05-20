## 📌 제작한 AI

| AI | 주요 역할 |
| --- | --- |
| 태그 추천 AI | 게시글 제목·본문에서 명사를 뽑아 어울리는 해시태그 3개 추천 |
| 조회수 예측 AI | 작성 중인 글과 비슷한 기존 게시글을 찾아 예상 조회수 산출 |
| 피드 챗봇 AI | 사이트 내 최근 피드를 근거로 RAG 답변, Redis 시맨틱 캐시 적용 |

## 🏷️ 태그 추천 AI

게시글 작성 시 제목과 본문을 분석해 어울리는 해시태그를 자동으로 제안하는 모델입니다. 사전에 학습된 분류 모델(`pkl/ad_tag_model.pkl`)과 LabelEncoder(`pkl/ad_tag_encoded.pkl`)를 가져와서, 형태소 분석으로 추출한 명사를 입력 피처로 사용합니다.

### 주요 기능

- Konlpy Okt 기반 한국어 명사 추출
- `predict_proba` 결과 정렬 후 상위 3개 태그 반환
- LabelEncoder를 이용한 클래스 → 태그 이름 매핑

### 핵심 코드

`post_contents = " ".join(self.okt.nouns(post_contents))`

`proba = self.model.predict_proba([post_contents])[0]`

## 📊 조회수 예측 AI

작성 중인 글과 가장 비슷한 기존 게시글들을 찾아 그 조회수를 가중평균하여 예상 조회수를 추정하는 모델입니다. 서버 부팅 시 `tbl_post`에서 1주일간 활성 피드를 모두 읽어와 TF-IDF 매트릭스를 미리 구축해 두고, 요청이 들어오면 새 글을 동일 공간으로 변환해 코사인 유사도를 계산합니다.

### 주요 기능

- FASTAPI 부팅시 active 상태의 게시글을 DB에서 로드한 뒤 TFIDF 벡터화
- 본문 + 태그를 합쳐 명사 단위로 토큰화
- 코사인 유사도 상위 5개 게시글 추출
- 유사도 점수를 가중치로 한 조회수 가중평균
- 예상 조회수, 최대/최소 조회수 함께 반환

### 핵심 코드

`sim_scores = cosine_similarity(new_post_matrix, self.tfidf_matrix)`

`predicted_view_count = np.dot(new_post_sim_scores, sim_view_counts) / new_post_sim_scores.sum()`

## 💬 피드 챗봇 AI

사이트에 올라온 최근 1주일간 피드를 활용해 사용자의 질문에 답하는 RAG 챗봇입니다. LangChain + FAISS로 벡터스토어를 구성하고, Redis를 활용한 시맨틱 캐시로 비슷한 질문은 LLM 호출 없이 즉시 응답합니다.

### 주요 기능

- FASTAPI 부팅시 피드 본문을 `feed_contents.txt`로 저장
- `RecursiveCharacterTextSplitter`로 200자 청크 분할
- FAISS 벡터스토어 생성 및 retriever 구성
- HuggingFace 한국어 임베딩 모델(`jhgan/ko-sbert-nli`) 싱글톤 공유
- 친근한 구어체 응답을 유도하는 PromptTemplate 적용
- Redis 시맨틱 캐시로 유사 질문 캐시 히트 처리 (`cache_score_threshold` 이하)
- 캐시 미스 시 LLM 호출 후 질문·답변을 Redis에 저장
- 응답을 한 글자씩 sleep 출력해 콘솔에서 타이핑 효과 확인

### 핵심 코드

`results = vector_db.similarity_search_with_score(question, k=1)`

`result = await self.feed_chain.ainvoke(question)`

## 🧩 AI별 핵심 요약

| AI | 핵심 책임 |
| --- | --- |
| 태그 추천 | 형태소 분석 + 사전 학습 모델로 게시글 해시태그 자동 제안 |
| 조회수 예측 | TF-IDF 유사도 기반 가중평균으로 신규 게시글 조회수 추정 |
| 피드 챗봇 | FAISS RAG + Redis 시맨틱 캐시로 피드 기반 대화 응답 |
