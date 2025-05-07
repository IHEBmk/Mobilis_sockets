from server.models import Wilaya
from rest_framework import serializers

class WilayaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Wilaya
        fields = ['name','id']
    def validate(self, data):
        if not data.get('name') or not data.get('id'):
            raise serializers.ValidationError("fields missing")
        return data