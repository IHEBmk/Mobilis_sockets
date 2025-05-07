

from server.models import Commune
from rest_framework import serializers


class CommuneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Commune
        fields = ['name','wilaya','id']
        
    def validate(self, data):
        if not data.get('name') or not data.get('wilaya') or not data.get('id'):
            raise serializers.ValidationError("fields missing")
        return data