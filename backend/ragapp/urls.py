from django.urls import path
from .views import ChatBotView
from django.views.generic import TemplateView

urlpatterns = [
    path("", TemplateView.as_view(template_name="index.html")),
    path("api/query/", ChatBotView.as_view()),
]
