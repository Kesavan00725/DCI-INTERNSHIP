import json, faiss, numpy as np, os
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_IDS = [
    "F5mRW0jo-U4",
    "rHux0gMZ3Eg",
    "OTmQOjsl0eg"
]

def fetch_transcript(video_id):
    try:
        t = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x["text"] for x in t])
    except:
        return ""

def chunk(text, size=700):
    return [text[i:i+size] for i in range(0, len(text), size)]

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    meta, chunks = [], []

    for vid in VIDEO_IDS:
        text = fetch_transcript(vid)
        parts = chunk(text)
        for p in parts:
            meta.append({
                "video_id": vid,
                "youtube_url": f"https://www.youtube.com/watch?v={vid}",
                "text": p
            })
            chunks.append(p)

    emb = model.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, f"{OUTPUT_DIR}/faiss_index.index")
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "ragapp",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "ragapp" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]
