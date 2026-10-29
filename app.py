# app.py
"""
Flask API for Arabic-English RAG Chatbot
"""

from flask import Flask, request, Response
import json
from datetime import datetime

from db import get_collection, get_stats
from rag import query_rag, generate_embedding
from utils import detect_language
from config import UNIVERSITY_NAME_AR, UNIVERSITY_NAME_EN

app = Flask(__name__)
col = get_collection()


def json_response(data, status=200):
    """Return JSON response with UTF-8 encoding."""
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        status=status,
        mimetype="application/json; charset=utf-8"
    )


@app.get("/")
def home():
    """Home endpoint with API information."""
    return json_response({
        "service": "Yarmouk University Arabic-English RAG Chatbot",
        "university_ar": UNIVERSITY_NAME_AR,
        "university_en": UNIVERSITY_NAME_EN,
        "version": "2.0.0",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check and statistics",
            "POST /query": "Ask questions (Arabic or English)",
            "GET /documents": "List documents",
            "GET /search": "Search documents by URL",
            "POST /embeddings/generate": "Generate embeddings for new documents"
        },
        "supported_languages": ["ar", "en"],
        "note": "Connects to existing MongoDB from scraping project"
    })


@app.get("/health")
def health():
    """Health check with database statistics."""
    try:
        stats = get_stats(col)

        # Check if RAG is ready
        rag_ready = stats["with_embeddings"] > 0

        return json_response({
            "status": "ok" if rag_ready else "needs_embeddings",
            "service": "Yarmouk University RAG Chatbot",
            "database": {
                "total_documents": stats["total_documents"],
                "with_embeddings": stats["with_embeddings"],
                "without_embeddings": stats["without_embeddings"],
                "html_documents": stats["html_documents"],
                "pdf_documents": stats["pdf_documents"],
                "completion": f"{stats['completion_percentage']}%"
            },
            "rag_ready": rag_ready,
            "message": "Ready to answer questions" if rag_ready else "Please generate embeddings first",
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return json_response({
            "status": "error",
            "error": str(e)
        }, 500)


@app.post("/query")
def query():
    """
    Main RAG query endpoint (supports Arabic and English).

    Request JSON:
    {
        "query": "ما هي شروط القبول؟" or "What are the admission requirements?",
        "top_k": 5,          // optional
        "min_score": 0.25    // optional
    }
    """
    try:
        data = request.get_json(silent=True) or {}
        user_query = data.get("query", "").strip()
        top_k = int(data.get("top_k", 5))
        min_score = float(data.get("min_score", 0.25))

        if not user_query:
            return json_response({
                "error": "missing query",
                "error_ar": "الرجاء إدخال سؤال",
                "error_en": "Please provide a query"
            }, 400)

        # Check if embeddings exist
        stats = get_stats(col)
        if stats["with_embeddings"] == 0:
            return json_response({
                "error": "No embeddings found",
                "error_ar": "لم يتم العثور على أي تضمينات. الرجاء تشغيل generate_embeddings.py أولاً",
                "error_en": "No embeddings found. Please run generate_embeddings.py first",
                "action": "Run: python generate_embeddings.py"
            }, 503)

        # Run RAG pipeline
        result = query_rag(col, user_query, top_k, min_score)
        result["timestamp"] = datetime.utcnow().isoformat()

        return json_response(result)

    except Exception as e:
        return json_response({
            "error": str(e),
            "error_ar": f"خطأ: {str(e)}",
            "error_en": f"Error: {str(e)}"
        }, 500)


@app.get("/documents")
def list_documents():
    """
    List documents from database.

    Query params:
        limit: max documents (default 10, max 50)
        skip: offset (default 0)
        has_embedding: filter by embedding status (true/false)
    """
    try:
        limit = int(request.args.get("limit", "10"))
        skip = int(request.args.get("skip", "0"))
        has_embedding = request.args.get("has_embedding", "").lower()

        # Validate limits
        limit = max(1, min(limit, 50))
        skip = max(0, skip)

        # Build filter
        filter_query = {}
        if has_embedding == "true":
            filter_query["embedding"] = {"$exists": True}
        elif has_embedding == "false":
            filter_query["embedding"] = {"$exists": False}

        # Query documents
        cursor = col.find(
            filter_query,
            {"_id": 0, "embedding": 0}  # Exclude _id and embedding
        ).skip(skip).limit(limit)

        docs = list(cursor)

        # Add language detection
        for doc in docs:
            content = doc.get("content", "")
            doc["detected_language"] = detect_language(content)

        total = col.count_documents(filter_query)

        return json_response({
            "total": total,
            "returned": len(docs),
            "limit": limit,
            "skip": skip,
            "documents": docs
        })

    except Exception as e:
        return json_response({"error": str(e)}, 500)


@app.get("/search")
def search_by_url():
    """
    Search for a specific document by URL pattern.

    Query params:
        url: URL or URL pattern to search
    """
    try:
        url_pattern = request.args.get("url", "").strip()

        if not url_pattern:
            return json_response({
                "error": "missing url parameter",
                "error_ar": "الرجاء إدخال عنوان URL",
                "error_en": "Please provide a URL"
            }, 400)

        # Search for URL (partial match)
        docs = list(col.find(
            {"url": {"$regex": url_pattern, "$options": "i"}},
            {"_id": 0, "embedding": 0}
        ).limit(10))

        return json_response({
            "query": url_pattern,
            "found": len(docs),
            "documents": docs
        })

    except Exception as e:
        return json_response({"error": str(e)}, 500)


@app.post("/embeddings/generate")
def generate_embeddings():
    """
    Generate embeddings for documents that don't have them.

    This is an alternative to running generate_embeddings.py
    Useful for API-based workflows.
    """
    try:
        # Count documents without embeddings
        docs_without = col.count_documents({"embedding": {"$exists": False}})

        if docs_without == 0:
            return json_response({
                "message": "All documents already have embeddings",
                "message_ar": "جميع الوثائق لديها تضمينات بالفعل",
                "message_en": "All documents already have embeddings",
                "processed": 0
            })

        # Find documents without embeddings
        cursor = col.find(
            {"embedding": {"$exists": False}},
            {"_id": 1, "content": 1, "url": 1}
        ).limit(100)  # Limit to 100 per request to avoid timeout

        processed = 0
        errors = 0

        for doc in cursor:
            try:
                content = doc.get("content", "")
                if not content:
                    errors += 1
                    continue

                # Generate embedding
                embedding = generate_embedding(content)
                if not embedding:
                    errors += 1
                    continue

                # Update document
                col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": embedding}}
                )
                processed += 1

            except Exception:
                errors += 1

        return json_response({
            "message": "Embeddings generated successfully",
            "message_ar": "تم توليد التضمينات بنجاح",
            "message_en": "Embeddings generated successfully",
            "processed": processed,
            "errors": errors,
            "remaining": docs_without - processed
        })

    except Exception as e:
        return json_response({
            "error": str(e),
            "error_ar": f"خطأ: {str(e)}",
            "error_en": f"Error: {str(e)}"
        }, 500)


@app.get("/stats")
def statistics():
    """Get detailed statistics about the database."""
    try:
        stats = get_stats(col)

        # Get language distribution
        docs = list(col.find({"content": {"$exists": True}}, {"content": 1, "_id": 0}).limit(100))

        arabic_count = 0
        english_count = 0
        for doc in docs:
            lang = detect_language(doc.get("content", ""))
            if lang == "ar":
                arabic_count += 1
            else:
                english_count += 1

        return json_response({
            "database": stats,
            "language_distribution_sample": {
                "arabic_documents": arabic_count,
                "english_documents": english_count,
                "sample_size": len(docs)
            },
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return json_response({"error": str(e)}, 500)


if __name__ == "__main__":
    print(f"""
╔════════════════════════════════════════════════════════════╗
║     {UNIVERSITY_NAME_AR}                              ║
║     {UNIVERSITY_NAME_EN}                          ║
║     Arabic-English RAG Chatbot API                        ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Show database stats on startup
    try:
        stats = get_stats(col)
        print(f"Database: {stats['total_documents']} documents")
        print(f"With embeddings: {stats['with_embeddings']}")
        print(f"Completion: {stats['completion_percentage']}%")

        if stats['with_embeddings'] == 0:
            print("\n⚠️  No embeddings found! Run: python generate_embeddings.py\n")
        else:
            print("\n✓ RAG system ready!\n")
    except:
        print("\n⚠️  Could not connect to database\n")

    app.run(host="127.0.0.1", port=5001, debug=True)
