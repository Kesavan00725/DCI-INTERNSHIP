import requests
from django.conf import settings

def ask_gpt(prompt):
    try:
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        body = {
            "model": settings.OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }

        r = requests.post(url, json=body, headers=headers)
        data = r.json()

        if "error" in data:
            return f"OpenAI Error: {data['error']['message']}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error contacting OpenAI: {str(e)}"
