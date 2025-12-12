from rest_framework.views import APIView
from rest_framework.response import Response
from .retriever import retrieve
from .openai_client import ask_gpt

INTRO = "Django is a high-level Python framework used for building secure and scalable applications."

class ChatBotView(APIView):
    def post(self, request):
        q = request.data.get("q", "")
        sources = retrieve(q)

        context = "\n\n".join([s["text"] for s in sources])
        prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {q}"

        answer = ask_gpt(prompt)

        videos = [
            "https://www.youtube.com/watch?v=F5mRW0jo-U4",
            "https://www.youtube.com/watch?v=rHux0gMZ3Eg",
            "https://www.youtube.com/watch?v=OTmQOjsl0eg"
        ]

        return Response({
            "intro": INTRO,
            "answer": answer,
            "videos": videos,
            "sources": sources
        })
