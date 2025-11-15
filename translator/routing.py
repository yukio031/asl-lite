# routing.py
from django.urls import path
from .consumers import ASLConsumer

websocket_urlpatterns = [
    path('ws/translate/', ASLConsumer.as_asgi()),  # Use path() instead of re_path()
]
