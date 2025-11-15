# routing.py
from django.urls import re_path
from .consumers import ASLConsumer

websocket_urlpatterns = [
    re_path(r'ws/translate/$', ASLConsumer.as_asgi()),
]
