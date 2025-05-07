# server/consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer

class UserLocationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = "user_location_room"  # Can be modified based on your needs
        self.room_group_name = f'location_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)

        user_name = text_data_json['name']
        latitude = text_data_json['latitude']
        longitude = text_data_json['longitude']

        # Send data to room group (for all web clients)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'send_user_location',
                'name': user_name,
                'latitude': latitude,
                'longitude': longitude
            }
        )

    async def send_user_location(self, event):
        user_name = event['name']
        latitude = event['latitude']
        longitude = event['longitude']

        # Send location data to WebSocket (web or mobile clients)
        await self.send(text_data=json.dumps({
            'name': user_name,
            'latitude': latitude,
            'longitude': longitude
        }))
