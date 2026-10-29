import requests
import json

BASE_URL = "http://127.0.0.1:5001"


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_health():
    """Test health endpoint."""
    print_section("فحص الصحة - Health Check")

    response = requests.get(f"{BASE_URL}/health")
    data = response.json()

    print(json.dumps(data, ensure_ascii=False, indent=2))

    if data.get("rag_ready"):
        print("\n✓ النظام جاهز - System ready!")
    else:
        print("\n⚠️  النظام غير جاهز - System not ready")
        print("قم بتشغيل - Run: python generate_embeddings.py")


def test_query(query_text):
    """Test RAG query."""
    print_section(f"اختبار السؤال - Testing Query")
    print(f"السؤال - Question: {query_text}")

    payload = {"query": query_text}

    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

    if response.status_code != 200:
        print(f"خطأ - Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()

    # Get language-specific labels
    labels = data.get("labels", {})

    print(f"\n{labels.get('query', 'Query')}: {data.get('query', '')}")
    print(
        f"{labels.get('confidence', 'Confidence')}: {data.get('confidence', {}).get(data.get('query_language', 'en'), 'N/A')}")
    print(f"\n{labels.get('answer', 'Answer')}:")
    print(data.get('answer', 'No answer'))

    sources = data.get('sources', [])
    if sources:
        print(f"\n{labels.get('sources', 'Sources')} ({len(sources)}):")
        for i, source in enumerate(sources, 1):
            print(f"\n  [{i}] {source.get('title', 'بدون عنوان - No title')}")
            print(f"      URL: {source.get('url', 'N/A')}")
            print(f"      Score: {source.get('score', 'N/A')}")
            lang = source.get('language', 'unknown')
            lang_label = "عربي" if lang == "ar" else "English"
            print(f"      Language: {lang_label}")


def test_documents():
    """Test document listing."""
    print_section("قائمة الوثائق - Document List")

    response = requests.get(f"{BASE_URL}/documents?limit=5")
    data = response.json()

    print(f"إجمالي الوثائق - Total: {data.get('total', 0)}")
    print(f"المعروضة - Returned: {data.get('returned', 0)}")

    docs = data.get('documents', [])
    if docs:
        print(f"\nأول {len(docs)} وثائق - First {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            lang = doc.get('detected_language', 'unknown')
            lang_label = "عربي" if lang == "ar" else "English"
            print(f"\n  {i}. {doc.get('title', 'بدون عنوان - No title')}")
            print(f"     URL: {doc.get('url', 'N/A')}")
            print(f"     النوع - Type: {doc.get('type', 'N/A')}")
            print(f"     اللغة - Language: {lang_label}")


def test_stats():
    """Test statistics endpoint."""
    print_section("الإحصائيات - Statistics")

    response = requests.get(f"{BASE_URL}/stats")
    data = response.json()

    print(json.dumps(data, ensure_ascii=False, indent=2))


def interactive_mode():
    """Interactive chat mode."""
    print_section("الوضع التفاعلي - Interactive Mode")
    print("اسأل أسئلة عن جامعة اليرموك - Ask questions about Yarmouk University")
    print("اكتب 'خروج' أو 'exit' للإنهاء - Type 'خروج' or 'exit' to quit\n")

    # Check health
    health_response = requests.get(f"{BASE_URL}/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        if not health_data.get("rag_ready"):
            print("⚠️  تحذير: النظام غير جاهز - Warning: System not ready")
            print("قم بتشغيل - Run: python generate_embeddings.py\n")

    while True:
        try:
            query = input("\nسؤالك - Your question: ").strip()

            if query.lower() in ['exit', 'quit', 'خروج', 'إنهاء']:
                print("\nمع السلامة - Goodbye!")
                break

            if not query:
                continue

            print("\nجاري التفكير - Thinking...")

            response = requests.post(
                f"{BASE_URL}/query",
                json={"query": query},
                headers={"Content-Type": "application/json; charset=utf-8"}
            )

            if response.status_code != 200:
                print(f"خطأ - Error: {response.status_code}")
                print(response.text)
                continue

            data = response.json()
            labels = data.get("labels", {})

            print(f"\n{'─' * 70}")
            print(f"{labels.get('answer', 'Answer')}:\n")
            print(data.get('answer', 'No answer'))

            confidence = data.get('confidence', {})
            lang = data.get('query_language', 'en')
            print(f"\n{labels.get('confidence', 'Confidence')}: {confidence.get(lang, 'N/A')}")

            sources = data.get('sources', [])
            if sources:
                print(f"\n{labels.get('sources', 'Sources')}:")
                for i, source in enumerate(sources, 1):
                    print(f"  [{i}] {source.get('url', 'N/A')} (Score: {source.get('score', 'N/A')})")

            print(f"{'─' * 70}")

        except KeyboardInterrupt:
            print("\n\nمع السلامة - Goodbye!")
            break
        except Exception as e:
            print(f"خطأ - Error: {str(e)}")


def run_comprehensive_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("جامعة اليرموك - Yarmouk University")
    print("اختبار شامل لنظام RAG - Comprehensive RAG System Test")
    print("=" * 70)

    # Test 1: Health
    test_health()

    # Test 2: Stats
    test_stats()

    # Test 3: Documents
    test_documents()

    # Test 4: Arabic queries
    print("\n" + "=" * 70)
    print("اختبار الأسئلة العربية - Testing Arabic Queries")
    print("=" * 70)

    arabic_questions = [
        "ما هي شروط القبول في الجامعة؟",
        "أخبرني عن قسم علوم الحاسوب",
        "ما هي البرامج الأكاديمية المتاحة؟"
    ]

    for q in arabic_questions:
        test_query(q)

    # Test 5: English queries
    print("\n" + "=" * 70)
    print("اختبار الأسئلة الإنجليزية - Testing English Queries")
    print("=" * 70)

    english_questions = [
        "What are the admission requirements?",
        "Tell me about the Computer Science department",
        "What academic programs are available?"
    ]

    for q in english_questions:
        test_query(q)

    print("\n" + "=" * 70)
    print("اكتمل الاختبار - Testing Complete!")
    print("=" * 70)


def main():
    """Main function."""
    import sys

    print("""
╔════════════════════════════════════════════════════════════════════╗
║              جامعة اليرموك - Yarmouk University                  ║
║         Arabic-English RAG Chatbot Test Client                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_comprehensive_tests()
        elif sys.argv[1] == "health":
            test_health()
        elif sys.argv[1] == "stats":
            test_stats()
        else:
            print(f"أمر غير معروف - Unknown command: {sys.argv[1]}")
            print("الاستخدام - Usage: python test_chatbot.py [test|health|stats|interactive]")
    else:
        # Default: interactive mode
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nتم المقاطعة - Interrupted")
    except Exception as e:
        print(f"\nخطأ - Error: {str(e)}")