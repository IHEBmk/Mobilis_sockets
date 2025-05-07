
from rest_framework import serializers
from .models import PointOfSale

class PointOfSaleSerializer(serializers.ModelSerializer):
    class Meta:
        model = PointOfSale
        fields = ['latitude', 'longitude', 'commune','id','created_at','status']
        
    def validate(self, data):
        # Optional: add custom validation
        if not data.get('latitude') or not data.get('longitude') or not data.get('commune') or not data.get('id'):
            raise serializers.ValidationError("fields missing")
        if not data.get('status'):
            data['status']=1
        return data
    
