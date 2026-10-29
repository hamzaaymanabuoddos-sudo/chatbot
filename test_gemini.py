# test_gemini.py
"""
Test Google Gemini API Key
"""

import google.generativeai as genai

# ضع مفتاحك هنا
API_KEY = "AIzaSyBTHORG2ODty-fif3_8tg2KLIHM7KdwKYQ"  # استبدل بمفتاحك الحقيقي

# Configure
genai.configure(api_key=API_KEY)

print("=" * 60)
print("اختبار Gemini API Key")
print("Testing Gemini API Key")
print("=" * 60)

# List available models
print("\n1. النماذج المتاحة - Available Models:")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"   ✓ {model.name}")
except Exception as e:
    print(f"   ❌ خطأ - Error: {e}")
    print("\n⚠️  تحقق من المفتاح - Check your API key!")
    exit(1)

# Try to generate content
print("\n2. اختبار توليد النص - Testing Text Generation:")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("مرحبا، ما هو اسمك؟")
    print(f"   ✓ الإجابة - Response: {response.text}")
    print("\n✅ المفتاح يعمل بنجاح! - API Key works!")
except Exception as e:
    print(f"   ❌ خطأ - Error: {e}")

    # Try older model
    print("\n   جرب نموذج أقدم - Trying older model...")
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello, what is your name?")
        print(f"   ✓ الإجابة - Response: {response.text}")
        print("\n✅ المفتاح يعمل! استخدم gemini-pro")
        print("✅ API Key works! Use gemini-pro")
    except Exception as e2:
        print(f"   ❌ خطأ - Error: {e2}")
        print("\n⚠️  المفتاح لا يعمل - API Key doesn't work!")
        print("تأكد من:")
        print("1. المفتاح صحيح من https://aistudio.google.com/app/apikey")
        print("2. تم تفعيل Gemini API")
        print("3. لديك حصة مجانية متبقية")