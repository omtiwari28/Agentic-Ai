import google.generativeai as genai
import os

# Set the key directly or from env
# I will use the key the user pasted in app.py
os.environ["GOOGLE_API_KEY"] = "AIzaSyBxrAyChQ-7hYq7wOJJSMnzcgz1Tp4bFcg"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("List of available models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
