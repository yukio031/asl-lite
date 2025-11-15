# myproject/routing.py
from django.urls import re_path
from translator import consumers

websocket_urlpatterns = [
    re_path(r'ws/translate/$', consumers.ASLConsumer.as_asgi()),
]
