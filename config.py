# config.py
"""
Configuration for Arabic-English RAG Chatbot
Uses Google Gemini (FREE API)
"""

# ============================================================================
# MONGODB CONNECTION (Connect to your existing database)
# ============================================================================
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "cs492_m1"  # Your existing database
COLLECTION_NAME = "documents"  # Your existing collection

# ============================================================================
# GOOGLE GEMINI CONFIGURATION (FREE!)
# ============================================================================
# Get your free API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyBtHH5pFtA0RkHBkY9cgmr8uz3TRkX95tU"  # ضع مفتاحك هنا

# Gemini Model - Use correct name!
GEMINI_MODEL = "gemini-2.5-flash"  # ✅ الاسم الصحيح

# ============================================================================
# EMBEDDING MODEL (Arabic-English Multilingual)
# ============================================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# ============================================================================
# RAG PARAMETERS
# ============================================================================
MAX_CONTEXT_DOCS = 8
MIN_SIMILARITY_SCORE = 0.15
MAX_EMBEDDING_LENGTH = 1500

# ============================================================================
# LLM PARAMETERS
# ============================================================================
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 800

# ============================================================================
# LANGUAGE SETTINGS
# ============================================================================
DEFAULT_LANGUAGE = "ar"
SUPPORT_BILINGUAL = True
UNIVERSITY_NAME_AR = "جامعة اليرموك"
UNIVERSITY_NAME_EN = "Yarmouk University"
UNIVERSITY_DOMAIN = "yu.edu.jo"