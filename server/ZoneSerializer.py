from server.models import Zone
from rest_framework import serializers

class ZoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Zone
        fields = ['commune','id','created_at']
    def validate(self, data):
        if not data.get('commune') or not data.get('id') or not data.get('created_at'):
            raise serializers.ValidationError("fields missing")
        return data