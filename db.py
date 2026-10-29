# db.py
"""
Database connection for Arabic RAG Chatbot
Connects to existing MongoDB database
"""

from pymongo import MongoClient, ASCENDING
from config import MONGO_URI, DB_NAME, COLLECTION_NAME


def get_collection():
    """
    Get MongoDB collection (connects to existing database).

    Returns:
        MongoDB collection object
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]

    # Create index for embeddings (improves query performance)
    try:
        col.create_index([("embedding", ASCENDING)])
    except Exception:
        pass  # Index might already exist

    return col


def get_stats(col):
    """
    Get database statistics.

    Args:
        col: MongoDB collection

    Returns:
        Dictionary with statistics
    """
    total_docs = col.count_documents({})
    with_embeddings = col.count_documents({"embedding": {"$exists": True}})
    without_embeddings = total_docs - with_embeddings

    # Count by type
    html_docs = col.count_documents({"type": "html"})
    pdf_docs = col.count_documents({"type": "pdf"})

    return {
        "total_documents": total_docs,
        "with_embeddings": with_embeddings,
        "without_embeddings": without_embeddings,
        "html_documents": html_docs,
        "pdf_documents": pdf_docs,
        "completion_percentage": round((with_embeddings / total_docs * 100), 2) if total_docs > 0 else 0
    }
#for test
from db import get_collection

col = get_collection()
print("Collection ready:", col.name)


