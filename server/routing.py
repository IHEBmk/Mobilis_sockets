# server/routing.py
from django.urls import re_path
from .consumers import UserLocationConsumer  # Assuming your consumer is named this

websocket_urlpatterns = [
    re_path(r'ws/location/$', UserLocationConsumer.as_asgi()),  # WebSocket URL pattern for location
]
