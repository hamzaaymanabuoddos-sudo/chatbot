# rag.py
"""
Hybrid RAG with Semantic + Keyword Retrieval and LLM Fallback
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re

from config import (
    EMBEDDING_MODEL, GEMINI_API_KEY,
    MAX_CONTEXT_DOCS, MIN_SIMILARITY_SCORE,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    UNIVERSITY_NAME_AR, UNIVERSITY_NAME_EN
)
from utils import detect_language, clean_text, format_response

# Initialize
genai.configure(api_key=GEMINI_API_KEY)
_embedding_model = None

# ─── Confidence Thresholds ────────────────────────────────────────────────────
# STRONG  : Context is reliable → use it exclusively
# MODERATE: Context is partial  → blend context + LLM knowledge
# WEAK    : Context is poor     → rely mostly on LLM, cite context if anything
THRESHOLD_STRONG   = 0.50
THRESHOLD_MODERATE = 0.30   # == MIN_SIMILARITY_SCORE by default

# ─── Domain Control ───────────────────────────────────────────────────────────
# Minimum semantic similarity to a "Yarmouk anchor" sentence required
# before we treat the query as in-domain.
DOMAIN_SEMANTIC_THRESHOLD = 0.30

# Hard keyword lists — presence of ANY item means instant in-domain.
DOMAIN_KEYWORDS_AR = {
    "يرموك", "اليرموك", "إربد",
    "كلية", "تسجيل", "قبول", "رسوم", "منح", "مكتبة",
    "أكاديمي", "فصل", "مساق", "شعبة", "خطة", "تخرج",
    "عمادة", "رئاسة", "دكتوراه", "ماجستير", "بكالوريوس",
    "طالب", "طلاب", "أستاذ", "محاضر", "مرشد",
    "نظام", "بوابة", "بريد", "جامعي",
}
DOMAIN_KEYWORDS_EN = {
    "yarmouk", "irbid", "yu",
    "faculty", "college", "registration", "admission", "fees", "scholarship",
    "academic", "semester", "course", "section", "plan", "graduation",
    "deanship", "presidency", "phd", "master", "bachelor",
    "student", "students", "professor", "lecturer", "advisor",
    "portal", "university email", "campus",
}

# Sentence used as the semantic anchor for domain detection.
# The query embedding is compared against this embedding to measure
# how "university-flavoured" the question is.
_DOMAIN_ANCHOR_TEXT = (
    "Yarmouk University admission, registration, fees, programs, "
    "faculty, students, academic calendar, campus, irbid jordan "
    "جامعة اليرموك القبول التسجيل الرسوم البرامج الكليات الطلاب"
)
_domain_anchor_embedding: List[float] = []   # cached on first use


# ─── Fixed Out-of-Domain Responses ───────────────────────────────────────────
OUT_OF_DOMAIN_RESPONSE = {
    "ar": (
        "عذراً، هذا المساعد مخصص للإجابة عن أسئلة جامعة اليرموك فقط. "
        "يُرجى طرح سؤال يتعلق بالجامعة (القبول، التسجيل، الرسوم، البرامج، إلخ)."
    ),
    "en": (
        "Sorry, this assistant only answers questions related to Yarmouk University. "
        "Please ask about the university (admission, registration, fees, programs, etc.)."
    ),
}


def get_embedding_model() -> SentenceTransformer:
    """Load embedding model (cached)."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✓ Model loaded successfully")
    return _embedding_model


def generate_embedding(text: str, max_length: int = 1500) -> List[float]:
    """Generate embedding for text."""
    if not text:
        return []
    text = clean_text(text)[:max_length]
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity."""
    if not vec1 or not vec2:
        return 0.0
    return float(np.dot(np.array(vec1), np.array(vec2)))


def extract_keywords(query: str) -> List[str]:
    """Extract keywords from query (Arabic + English)."""
    stop_words = {
        'ما', 'هي', 'هو', 'في', 'على', 'من', 'إلى', 'عن', 'كيف', 'ماذا', 'أين',
        'التي', 'الذي', 'هل', 'كم', 'متى', 'لماذا', 'أن', 'إن', 'لا', 'نعم',
        'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for',
        'how', 'where', 'when', 'why', 'do', 'does', 'can', 'could', 'would'
    }
    words = re.findall(r'\b\w+\b', query.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


def keyword_score(query: str, content: str) -> float:
    """Calculate keyword matching score."""
    keywords = extract_keywords(query)
    content_lower = content.lower()
    if not keywords:
        return 0.0
    matches = sum(1 for kw in keywords if kw in content_lower)
    return matches / len(keywords)


def expand_query(query: str, query_lang: str) -> List[str]:
    """Expand query with synonyms."""
    expansions = [query]

    if query_lang == "ar":
        synonyms = {
            "شروط": ["متطلبات", "معايير", "ضوابط"],
            "القبول": ["الالتحاق", "التسجيل", "الانضمام"],
            "الرسوم": ["التكاليف", "المصاريف", "الأسعار"],
            "برامج": ["تخصصات", "أقسام", "مجالات"],
            "كلية": ["قسم", "faculty", "college"],
            "جامعة": ["university", "الجامعة"],
            "طالب": ["student", "دارس"],
            "دراسة": ["تعليم", "تدريس"]
        }
    else:
        synonyms = {
            "admission": ["enrollment", "acceptance", "entry"],
            "requirements": ["criteria", "conditions", "prerequisites"],
            "fees": ["tuition", "cost", "charges"],
            "programs": ["majors", "departments", "courses"],
            "faculty": ["college", "department", "school"],
            "student": ["learner", "pupil"],
            "university": ["college", "institution"]
        }

    query_lower = query.lower()
    for word, syns in synonyms.items():
        if word in query_lower:
            for syn in syns:
                expanded = query_lower.replace(word, syn)
                if expanded not in expansions:
                    expansions.append(expanded)

    return expansions[:3]


# ─── Domain Detection ────────────────────────────────────────────────────────

def is_in_domain(query: str) -> bool:
    """
    Determine whether the query is related to Yarmouk University.

    Two-stage check (fast → slow):

    Stage 1 — Keyword match  (zero cost, runs first)
        If ANY domain keyword appears in the query → immediately in-domain.
        This handles explicit mentions like "يرموك", "Yarmouk", "YU fees", etc.

    Stage 2 — Semantic similarity  (embedding cost, runs only if Stage 1 misses)
        Embed the query and compare against a pre-built "domain anchor" embedding
        that describes Yarmouk University topics.
        If cosine_similarity >= DOMAIN_SEMANTIC_THRESHOLD → in-domain.
        This catches implicit questions like "ما هي رسوم الساعة الدراسية؟"
        that mention no explicit university name but are clearly about a university.

    Returns True  → process the query normally through RAG.
    Returns False → skip RAG + LLM, return out-of-domain refusal.

    Design notes:
    - We intentionally keep DOMAIN_SEMANTIC_THRESHOLD low (0.30) to avoid
      false rejections; the keyword list handles high-precision cases.
    - "جامعة" / "university" alone are NOT in the keyword lists because a user
      could ask "ما هو أفضل جامعة في الأردن؟" which is out-of-domain.
      Such cases fall through to the semantic check.
    """
    global _domain_anchor_embedding

    query_lower = query.lower()

    # ── Stage 1: Keyword match ────────────────────────────────────────────────
    for kw in DOMAIN_KEYWORDS_AR:
        if kw in query:          # Arabic: no .lower() needed (already cased)
            return True
    for kw in DOMAIN_KEYWORDS_EN:
        if kw in query_lower:
            return True

    # ── Stage 2: Semantic similarity ─────────────────────────────────────────
    # Build anchor embedding once and cache it
    if not _domain_anchor_embedding:
        _domain_anchor_embedding = generate_embedding(_DOMAIN_ANCHOR_TEXT)

    query_embedding = generate_embedding(query)
    if not query_embedding or not _domain_anchor_embedding:
        # If embeddings fail for any reason, default to ALLOWING the query
        # (fail open) so the RAG pipeline can decide with retrieved docs.
        return True

    similarity = cosine_similarity(query_embedding, _domain_anchor_embedding)
    return similarity >= DOMAIN_SEMANTIC_THRESHOLD


def build_out_of_domain_response(query: str, query_lang: str) -> Dict:
    """Return a structured refusal response for out-of-domain queries."""
    message = OUT_OF_DOMAIN_RESPONSE.get(query_lang, OUT_OF_DOMAIN_RESPONSE["en"])
    response = {
        "answer": message,
        "sources": [],
        "confidence": {"ar": "غير قابل للتطبيق", "en": "N/A"},
        "context_quality": "out_of_domain",
        "query": query,
        "query_language": query_lang,
        "num_sources": 0,
        "in_domain": False,
    }
    return format_response(response, query_lang)


def retrieve_relevant_docs(col, query: str, top_k: int = 5, min_score: float = 0.25) -> List[Dict]:
    """
    Hybrid retrieval: Semantic (70%) + Keyword (30%).
    Returns top-k docs sorted by combined score.
    """
    query_lang = detect_language(query)
    queries = expand_query(query, query_lang)

    all_scored_docs = []
    seen_urls = set()

    for q in queries:
        query_embedding = generate_embedding(q)
        if not query_embedding:
            continue

        docs = list(col.find(
            {"embedding": {"$exists": True}},
            {"_id": 0, "url": 1, "title": 1, "content": 1, "type": 1, "embedding": 1}
        ))

        if not docs:
            continue

        for doc in docs:
            if "embedding" not in doc:
                continue

            url = doc.get("url", "")
            if url in seen_urls:
                continue

            content = doc.get("content", "")
            semantic_score = cosine_similarity(query_embedding, doc["embedding"])
            kw_score = keyword_score(q, content)
            combined_score = (0.7 * semantic_score) + (0.3 * kw_score)

            if combined_score >= min_score or semantic_score >= min_score or kw_score >= 0.3:
                all_scored_docs.append({
                    "url": url,
                    "title": doc.get("title", ""),
                    "type": doc.get("type", ""),
                    "content": content[:800],
                    "semantic_score": round(semantic_score, 3),
                    "keyword_score": round(kw_score, 3),
                    "combined_score": round(combined_score, 3),
                    "score": round(combined_score, 3),
                    "language": detect_language(content)
                })
                seen_urls.add(url)

    all_scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)
    return all_scored_docs[:top_k]


# ─── NEW: Context Quality Assessment ─────────────────────────────────────────

def assess_context_quality(docs: List[Dict]) -> str:
    """
    Classify retrieved context into three tiers:
      - 'strong'   : top doc score >= THRESHOLD_STRONG
      - 'moderate' : top doc score >= THRESHOLD_MODERATE
      - 'weak'     : below both thresholds or no docs

    Decision is driven by the BEST document, not the average,
    because even a single highly relevant doc is enough context.
    """
    if not docs:
        return "weak"

    best_score = docs[0].get("combined_score", docs[0].get("score", 0.0))

    if best_score >= THRESHOLD_STRONG:
        return "strong"
    elif best_score >= THRESHOLD_MODERATE:
        return "moderate"
    else:
        return "weak"


# ─── IMPROVED: Prompt Construction ───────────────────────────────────────────

def construct_prompt(query: str, context_docs: List[Dict], query_lang: str) -> str:
    """
    Build a Hybrid RAG prompt whose instructions adapt to context quality:
      - strong   → use context; do NOT add external knowledge
      - moderate → prefer context; fill gaps with general knowledge
      - weak     → use general knowledge; mention any weak context found
    """
    context_quality = assess_context_quality(context_docs)

    # Build context block
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        score_info = (
            f"تشابه دلالي: {doc.get('semantic_score', 'N/A')}, "
            f"تطابق كلمات: {doc.get('keyword_score', 'N/A')}, "
            f"النقاط المجمّعة: {doc.get('combined_score', 'N/A')}"
            if query_lang == "ar" else
            f"Semantic: {doc.get('semantic_score', 'N/A')}, "
            f"Keyword: {doc.get('keyword_score', 'N/A')}, "
            f"Combined: {doc.get('combined_score', 'N/A')}"
        )
        if query_lang == "ar":
            context_parts.append(
                f"[مصدر {i}] ({score_info})\n"
                f"العنوان: {doc['title']}\n"
                f"الرابط: {doc['url']}\n"
                f"المحتوى: {doc['content']}\n"
            )
        else:
            context_parts.append(
                f"[Source {i}] ({score_info})\n"
                f"Title: {doc['title']}\n"
                f"URL: {doc['url']}\n"
                f"Content: {doc['content']}\n"
            )

    context_text = "\n".join(context_parts) if context_parts else (
        "لا توجد وثائق مسترجعة." if query_lang == "ar" else "No documents retrieved."
    )

    # ── Arabic Prompts ────────────────────────────────────────────────────────
    if query_lang == "ar":

        if context_quality == "strong":
            behavior_instructions = """
سلوك الإجابة (السياق قوي — التزم بالوثائق):
- استخدم الوثائق المسترجعة حصرياً كمصدر لإجابتك.
- ابدأ إجابتك بـ "وفقاً لوثائق جامعة اليرموك..."
- اذكر رقم المصدر [مصدر N] بعد كل معلومة مباشرةً.
- لا تُضف أي معلومات من معرفتك العامة.
- إذا لم تُجِب الوثائق على جزء من السؤال، قل ذلك صراحةً.
"""
        elif context_quality == "moderate":
            behavior_instructions = """
سلوك الإجابة (السياق جزئي — مزج الوثائق مع المعرفة العامة):
- أعطِ الأولوية للمعلومات الواردة في الوثائق المسترجعة.
- يمكنك استكمال المعلومات الناقصة من معرفتك العامة عن جامعة اليرموك تحديداً إن لزم الأمر.
- إذا استخدمت الوثائق فقط، ابدأ بـ "وفقاً لوثائق جامعة اليرموك..."
- إذا دمجت الوثائق مع معرفتك العامة، ابدأ بـ "بناءً على بيانات الجامعة والمعرفة العامة..."
- اذكر رقم المصدر [مصدر N] لكل معلومة مأخوذة من الوثائق.
- وضّح بوضوح أي معلومة تأتي من معرفتك العامة وليس من الوثائق.
"""
        else:  # weak
            behavior_instructions = """
سلوك الإجابة (السياق ضعيف — اعتمد على المعرفة العامة عن جامعة اليرموك):
- الوثائق المسترجعة ذات صلة منخفضة أو غير متوفرة؛ لا تعتمد عليها كمرجع رئيسي.
- استخدم معرفتك عن جامعة اليرموك تحديداً والنظام الأكاديمي الأردني للإجابة.
- ابدأ إجابتك بـ "بناءً على المعلومات المتاحة عن جامعة اليرموك..."
- إذا كانت الوثائق تحتوي على أي معلومة وثيقة الصلة رغم ضعفها، اذكرها مع رقم المصدر.
- أوصِ المستخدم بالتحقق مباشرةً من المصادر الرسمية للجامعة للحصول على معلومات محدّثة.
"""

        prompt = f"""أنت مساعد ذكي مخصص حصراً لجامعة اليرموك (Yarmouk University). نطاق عملك محدود بشكل صارم بالأسئلة المتعلقة بجامعة اليرموك فقط.

قيود النطاق (مطلقة — لا استثناء):
- أجب فقط عن الأسئلة المتعلقة بجامعة اليرموك: القبول، التسجيل، الرسوم، البرامج، الكليات، الأنشطة، إلخ.
- إذا طُلب منك الإجابة عن جامعة أخرى أو موضوع لا علاقة له بجامعة اليرموك، قل فقط:
  "عذراً، هذا المساعد مخصص لجامعة اليرموك فقط."
- لا تُجب عن أسئلة عامة (علوم، أخبار، سياسة، ترفيه، إلخ) إلا إذا كانت مرتبطة مباشرةً بجامعة اليرموك.

{behavior_instructions}

قواعد عامة لا تتغير:
- لا تخترع أرقاماً أو تواريخ أو أسماء لم ترد في المصادر.
- استخدم نقاطاً (bullet points) إذا كانت الإجابة تحتوي على عناصر متعددة.
- حافظ على لغة عربية فصيحة ومهنية.
- لا تتجاهل أي معلومة ذات صلة موجودة في الوثائق.

الوثائق المسترجعة (جودة السياق: {context_quality}):
{context_text}

سؤال المستخدم: {query}

الإجابة:"""

    # ── English Prompts ───────────────────────────────────────────────────────
    else:

        if context_quality == "strong":
            behavior_instructions = """
ANSWER BEHAVIOR (Strong context — stay within documents):
- Use ONLY the retrieved documents as your source.
- Start your answer with "According to Yarmouk University documents, ..."
- Cite every piece of information with [Source N] immediately after it.
- Do NOT add external knowledge beyond what the documents provide.
- If the documents do not answer part of the question, state that clearly.
"""
        elif context_quality == "moderate":
            behavior_instructions = """
ANSWER BEHAVIOR (Partial context — blend documents with Yarmouk-specific knowledge):
- Prioritize information found in the retrieved documents.
- Fill in gaps using your knowledge specifically about Yarmouk University only when the documents are insufficient.
- If you use documents only → start with "According to Yarmouk University documents, ..."
- If you blend documents + general knowledge → start with "Based on university data and general knowledge, ..."
- Cite document-sourced facts with [Source N].
- Explicitly flag any information that comes from general knowledge rather than the documents.
"""
        else:  # weak
            behavior_instructions = """
ANSWER BEHAVIOR (Weak context — rely on Yarmouk-specific knowledge):
- Retrieved documents have low relevance or are absent; do not treat them as primary sources.
- Use your knowledge specifically about Yarmouk University and the Jordanian academic system.
- Start your answer with "Based on available information about Yarmouk University, ..."
- If any retrieved document contains a closely relevant fact despite low scores, include it with [Source N].
- Recommend that the user verify directly with official Yarmouk University sources for up-to-date information.
"""

        prompt = f"""You are an intelligent assistant dedicated exclusively to Yarmouk University (جامعة اليرموك). Your scope is strictly limited to questions about Yarmouk University only.

DOMAIN RESTRICTION (absolute — no exceptions):
- Only answer questions about Yarmouk University: admission, registration, fees, programs, faculties, activities, etc.
- If asked about any other university or topic unrelated to Yarmouk University, respond only with:
  "Sorry, this assistant is dedicated to Yarmouk University only."
- Do not answer general knowledge questions (science, news, politics, entertainment, etc.) unless they are directly related to Yarmouk University.

{behavior_instructions}

UNIVERSAL RULES (never change):
- Never fabricate numbers, dates, or names not present in the sources.
- Use bullet points when the answer contains multiple items.
- Use professional, clear language.
- Never ignore relevant information found in the documents.

RETRIEVED DOCUMENTS (context quality: {context_quality}):
{context_text}

USER QUESTION: {query}

ANSWER:"""

    return prompt


def generate_answer(prompt: str, temperature: float = LLM_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS) -> str:
    """Generate answer from Google Gemini."""
    try:
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config
        )

        response = model.generate_content(prompt)

        if response.text:
            return response.text.strip()
        else:
            if "ar" in prompt[:100]:
                return "المعذرة، لم أتمكن من توليد إجابة. يرجى إعادة صياغة السؤال."
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        error_msg = str(e)
        print(f"Gemini Error: {error_msg}")
        if "ar" in prompt[:100]:
            return f"خطأ في توليد الإجابة: {error_msg}"
        else:
            return f"Error generating response: {error_msg}"


# ─── UPDATED: Confidence Assessment ──────────────────────────────────────────

def assess_confidence(sources: List[Dict]) -> Dict[str, str]:
    """
    Map context quality tier to human-readable confidence labels.
    Uses the same tier logic as assess_context_quality() for consistency.
    """
    quality = assess_context_quality(sources)
    mapping = {
        "strong":   {"ar": "عالية",   "en": "High"},
        "moderate": {"ar": "متوسطة",  "en": "Medium"},
        "weak":     {"ar": "منخفضة",  "en": "Low"},
    }
    return mapping.get(quality, {"ar": "لا توجد بيانات", "en": "No Data"})


# ─── MAIN RAG PIPELINE ────────────────────────────────────────────────────────

def query_rag(col, query: str, top_k: int = MAX_CONTEXT_DOCS, min_score: float = MIN_SIMILARITY_SCORE) -> Dict:
    """
    Hybrid RAG pipeline:
      1. Retrieve top-k docs using hybrid search (semantic + keyword).
      2. Classify context quality (strong / moderate / weak).
      3. Build an adaptive prompt that instructs the LLM how much to rely
         on context vs. general knowledge based on quality tier.
      4. Generate answer with Gemini and return structured response.

    Pseudo-code summary:
      docs  = hybrid_retrieve(query, top_k, min_score)
      tier  = classify(best_doc_score)          # strong | moderate | weak
      prompt = build_prompt(query, docs, tier)  # instructions adapt to tier
      answer = llm(prompt)
      return {answer, sources, confidence, tier, ...}
    """
    query_lang = detect_language(query)

    # Step 0 – Domain gate (runs BEFORE any DB or LLM call)
    # If the query is unrelated to Yarmouk University, refuse immediately.
    if not is_in_domain(query):
        return build_out_of_domain_response(query, query_lang)

    # Step 1 – Retrieve
    relevant_docs = retrieve_relevant_docs(col, query, top_k, min_score)

    # Step 2 – Assess context quality
    context_quality = assess_context_quality(relevant_docs)

    # Step 3 – Handle completely empty results (no docs at all)
    # Even here we fall through to the LLM rather than returning a hard "no data"
    # so the model can answer from general knowledge.
    if not relevant_docs:
        rag_prompt = construct_prompt(query, [], query_lang)
        answer = generate_answer(rag_prompt)
        response = {
            "answer": answer,
            "sources": [],
            "confidence": assess_confidence([]),
            "context_quality": "weak",
            "query": query,
            "query_language": query_lang,
            "num_sources": 0,
            "in_domain": True,
        }
        return format_response(response, query_lang)

    # Step 4 – Build adaptive prompt
    rag_prompt = construct_prompt(query, relevant_docs, query_lang)

    # Step 5 – Generate answer
    answer = generate_answer(rag_prompt)

    # Step 6 – Build response
    response = {
        "answer": answer,
        "sources": relevant_docs,
        "confidence": assess_confidence(relevant_docs),
        "context_quality": context_quality,
        "query": query,
        "query_language": query_lang,
        "num_sources": len(relevant_docs),
        "in_domain": True,
    }

    return format_response(response, query_lang)
