# asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path
from translator import consumers
import translator.routing  # Correct import

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'asl_lite.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path('ws/translate/', consumers.ASLConsumer.as_asgi()),
        ])
    ),
})
