import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "change-me")
DEBUG = os.getenv("DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]

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

ROOT_URLCONF = "django_rag.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "ragapp" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "django_rag.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "/static/"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# RAG file paths
DEFAULT_INGEST_DIR = Path(__file__).resolve().parent.parent / "ingest"
_faiss_env = os.getenv("FAISS_INDEX")
_meta_env = os.getenv("META_JSON")
if _faiss_env:
    # Normalize relative env paths: strip leading 'backend/' if present
    faiss_text = _faiss_env.replace("\\", "/")
    if faiss_text.startswith("backend/"):
        faiss_text = faiss_text[len("backend/"):]
    faiss_path = Path(faiss_text)
    if not faiss_path.is_absolute():
        FAISS_INDEX = str(Path(__file__).resolve().parent.parent / faiss_path)
    else:
        FAISS_INDEX = str(faiss_path)
else:
    FAISS_INDEX = str(DEFAULT_INGEST_DIR / "faiss_index.index")

if _meta_env:
    # Normalize relative env paths: strip leading 'backend/' if present
    meta_text = _meta_env.replace("\\", "/")
    if meta_text.startswith("backend/"):
        meta_text = meta_text[len("backend/"):]
    meta_path = Path(meta_text)
    if not meta_path.is_absolute():
        META_JSON = str(Path(__file__).resolve().parent.parent / meta_path)
    else:
        META_JSON = str(meta_path)
else:
    META_JSON = str(DEFAULT_INGEST_DIR / "metadata.json")
