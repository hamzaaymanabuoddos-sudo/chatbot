
"""
Generate embeddings for all documents in MongoDB
Optimized for Arabic-English content
"""

from db import get_collection, get_stats
from rag import generate_embedding
from utils import detect_language
from tqdm import tqdm
import time


def generate_all_embeddings():
    """Generate embeddings for all documents without them."""
    col = get_collection()

    print("=" * 60)
    print("جامعة اليرموك - Yarmouk University")
    print("Arabic-English Embedding Generator")
    print("=" * 60)

    # Get statistics
    stats = get_stats(col)

    print(f"\nإحصائيات قاعدة البيانات - Database Statistics:")
    print(f"  إجمالي الوثائق - Total documents: {stats['total_documents']}")
    print(f"  مع التضمينات - With embeddings: {stats['with_embeddings']}")
    print(f"  بدون تضمينات - Without embeddings: {stats['without_embeddings']}")
    print(f"  وثائق HTML - HTML documents: {stats['html_documents']}")
    print(f"  وثائق PDF - PDF documents: {stats['pdf_documents']}")
    print(f"  نسبة الإكمال - Completion: {stats['completion_percentage']}%")
    print()

    if stats['without_embeddings'] == 0:
        print("✓ جميع الوثائق لديها تضمينات بالفعل")
        print("✓ All documents already have embeddings!")
        return

    # Find documents without embeddings
    cursor = col.find(
        {"embedding": {"$exists": False}},
        {"_id": 1, "url": 1, "content": 1, "title": 1}
    )

    docs = list(cursor)
    total = len(docs)

    print(f"معالجة {total} وثيقة - Processing {total} documents...")
    print()

    success = 0
    errors = 0
    arabic_docs = 0
    english_docs = 0

    # Process with progress bar
    with tqdm(total=total, desc="Generating embeddings", ncols=80) as pbar:
        for doc in docs:
            try:
                content = doc.get("content", "")

                if not content:
                    print(f"\n⚠️  Empty content: {doc.get('url', 'unknown')}")
                    errors += 1
                    pbar.update(1)
                    continue

                # Detect language
                lang = detect_language(content)
                if lang == "ar":
                    arabic_docs += 1
                else:
                    english_docs += 1

                # Generate embedding
                embedding = generate_embedding(content)

                if not embedding:
                    errors += 1
                    pbar.update(1)
                    continue

                # Update document
                col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "embedding": embedding,
                        "detected_language": lang
                    }}
                )

                success += 1
                pbar.update(1)

            except Exception as e:
                print(f"\n❌ Error: {doc.get('url', 'unknown')}: {str(e)}")
                errors += 1
                pbar.update(1)

    print()
    print("=" * 60)
    print("اكتمال المعالجة - Processing Complete!")
    print("=" * 60)
    print(f"✓ تمت المعالجة بنجاح - Successfully processed: {success}")
    print(f"❌ أخطاء - Errors: {errors}")
    print(f"📝 وثائق عربية - Arabic documents: {arabic_docs}")
    print(f"📝 وثائق إنجليزية - English documents: {english_docs}")
    print("=" * 60)


def verify_embeddings(sample_size=5):
    """Verify embeddings with language detection."""
    col = get_collection()

    print("=" * 60)
    print(f"التحقق من {sample_size} تضمينات عشوائية")
    print(f"Verifying {sample_size} Random Embeddings")
    print("=" * 60)

    docs = list(col.aggregate([
        {"$match": {"embedding": {"$exists": True}}},
        {"$sample": {"size": sample_size}}
    ]))

    if not docs:
        print("❌ لم يتم العثور على وثائق مع تضمينات")
        print("❌ No documents with embeddings found!")
        return

    for i, doc in enumerate(docs, 1):
        embedding = doc.get("embedding", [])
        content = doc.get("content", "")
        lang = detect_language(content)

        print(f"\n{i}. {doc.get('title', 'بدون عنوان - No title')}")
        print(f"   URL: {doc.get('url', 'N/A')}")
        print(f"   النوع - Type: {doc.get('type', 'N/A')}")
        print(f"   اللغة - Language: {'عربي - Arabic' if lang == 'ar' else 'English - إنجليزي'}")
        print(f"   طول المحتوى - Content length: {len(content)} chars")
        print(f"   بعد التضمين - Embedding dimension: {len(embedding)}")
        print(f"   أول 5 قيم - First 5 values: {embedding[:5]}")


def regenerate_all():
    """Regenerate ALL embeddings."""
    col = get_collection()

    print("\n⚠️  تحذير - WARNING!")
    print("سيتم إعادة توليد جميع التضمينات")
    print("This will regenerate ALL embeddings")

    response = input("\nهل أنت متأكد؟ اكتب 'نعم' أو 'yes' للمتابعة: ")

    if response.lower() not in ['yes', 'نعم']:
        print("تم الإلغاء - Cancelled")
        return

    print("\nإزالة التضمينات الموجودة - Removing existing embeddings...")
    result = col.update_many({}, {"$unset": {"embedding": "", "detected_language": ""}})
    print(f"تمت إزالة التضمينات من {result.modified_count} وثيقة")
    print(f"Removed embeddings from {result.modified_count} documents")

    generate_all_embeddings()


def main():
    """Main menu."""
    while True:
        print("\n" + "=" * 60)
        print("القائمة الرئيسية - Main Menu")
        print("=" * 60)
        print("1. توليد التضمينات للوثائق الجديدة - Generate embeddings for new documents")
        print("2. إظهار الإحصائيات - Show statistics")
        print("3. التحقق من التضمينات - Verify embeddings (sample)")
        print("4. إعادة توليد جميع التضمينات - Regenerate ALL embeddings (⚠️  destructive)")
        print("5. خروج - Exit")
        print()

        choice = input("اختر خياراً - Select option (1-5): ").strip()

        if choice == "1":
            start_time = time.time()
            generate_all_embeddings()
            elapsed = time.time() - start_time
            print(f"\nالوقت المستغرق - Time taken: {elapsed:.2f} seconds")

        elif choice == "2":
            col = get_collection()
            stats = get_stats(col)
            print(f"\nإجمالي الوثائق - Total: {stats['total_documents']}")
            print(f"مع التضمينات - With embeddings: {stats['with_embeddings']}")
            print(f"بدون تضمينات - Without: {stats['without_embeddings']}")
            print(f"وثائق HTML - HTML: {stats['html_documents']}")
            print(f"وثائق PDF - PDF: {stats['pdf_documents']}")
            print(f"نسبة الإكمال - Completion: {stats['completion_percentage']}%")

        elif choice == "3":
            verify_embeddings()

        elif choice == "4":
            regenerate_all()

        elif choice == "5":
            print("\nمع السلامة - Goodbye!")
            break

        else:
            print("خيار غير صحيح - Invalid option")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nتم المقاطعة بواسطة المستخدم - Interrupted by user")
    except Exception as e:
        print(f"\nخطأ - Error: {str(e)}")

