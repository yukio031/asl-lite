# translator/routing.py
from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path("ws/translate/", consumers.ASLConsumer.as_asgi()),
]
