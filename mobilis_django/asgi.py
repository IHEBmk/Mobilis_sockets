# asgi.py
import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import server.routing  # Ensure this imports the routing configuration correctly

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mobilis_django.settings")

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(
        server.routing.websocket_urlpatterns  # Uses your URL router for WebSockets
    ),
})
