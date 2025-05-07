# Userserializers.py
from datetime import datetime
from django.utils import timezone
import uuid
from rest_framework import serializers
from .models import Agence, User, Wilaya
import hashlib

class UserSignupSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['email', 'phone', 'password', 'role', 'wilaya','manager']
        extra_kwargs = {'password': {'write_only': True}}

    def validate(self, data):
        if not data.get('email') and not data.get('phone'):
            raise serializers.ValidationError("Either email or phone must be provided.")

        if len(data.get('password', '')) < 6:
            raise serializers.ValidationError("Password must be at least 6 characters long.")
        
        # Validate wilaya
        if data.get('wilaya'):
            try:
                data['wilaya'] = data.get('wilaya') 
            except ValueError:
                raise serializers.ValidationError("Invalid Wilaya UUID format.")
            except Wilaya.DoesNotExist:
                raise serializers.ValidationError("Wilaya not found.")
                
        # Validate manager
        if data.get('role') == 'agent':
            try:
                if data.get('manager')!=None:
                    manager = data.get('manager')
            except User.DoesNotExist:
                raise serializers.ValidationError("Manager does not exist.")
            if manager.role != 'manager':
                raise serializers.ValidationError("Manager must be a manager.")
            data['manager'] = manager
            print("here")
            print(data['manager'])
        data['created_at'] = datetime.now()
        data['id'] = uuid.uuid4()
                # Validate agence
        if data.get('agence'):
            try:
                data['agence'] = Agence.objects.get(id=data.get('agence'))
            except Agence.DoesNotExist:
                raise serializers.ValidationError("Agence not found.")

        # Check for existing users
        if data.get('email'):
            try:
                User.objects.get(email=data.get('email'))
                raise serializers.ValidationError("Email already exists.")
            except User.DoesNotExist:
                pass
                
        if data.get('phone'):
            try:
                User.objects.get(phone=data.get('phone'))
                raise serializers.ValidationError("Phone already exists.")
            except User.DoesNotExist:
                pass
                
        # Hash password for security
        data['password'] = hashlib.sha256(data['password'].encode()).hexdigest()
        
        return data

    def create(self, validated_data):
        user = User.objects.create(**validated_data)
        return user


class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField(required=False)
    phone = serializers.CharField(required=False)
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        email = data.get('email', None)
        phone = data.get('phone', None)
        password = data.get('password')

        if not (email or phone):
            raise serializers.ValidationError("Either email or phone is required.")
        
        try:
            if email:
                users = User.objects.filter(email=email)
            else:
                users =(User.objects.filter(phone=phone))
        except User.DoesNotExist:
            raise serializers.ValidationError("User not found.")
        
        # Verify hashed password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if not users:
            raise serializers.ValidationError("User not found.")
        for user in users:
            if user.password == hashed_password:
                user.last_seen = timezone.now()
                user.save()
                data['user'] = user
                return data
        raise serializers.ValidationError("Incorrect password.")


class UserAssignSerializer(serializers.Serializer):
    id = serializers.CharField(required=True)
    zone = serializers.CharField(required=True)
    assigner = serializers.CharField(required=True)
    
    def validate(self, data):
        id = data.get('id')
        zone = data.get('zone')
        assigner = data.get('assigner')
        
        # Call parent validation
        data = super().validate(data)
        
        try:
            user = User.objects.get(id=id)
        except User.DoesNotExist:
            raise serializers.ValidationError("User not found.")
            
        try:
            assigner_user = User.objects.get(id=assigner)
        except User.DoesNotExist:
            raise serializers.ValidationError("Assigner not found.")
            
        # Check permissions
        if assigner_user.role != '0' and (assigner_user.role == '1' and assigner_user.wilaya != user.wilaya):
            raise serializers.ValidationError("Assigner must be a manager.")
            
        # Store for use in save method
        data['user_obj'] = user
        data['zone_id'] = zone
        
        return data
    
    def save(self):
        user = self.validated_data['user_obj']
        from .models import Zone
        try:
            zone = Zone.objects.get(id=self.validated_data['zone_id'])
            user.zone = zone
            user.save(update_fields=['zone'])
            return user
        except Zone.DoesNotExist:
            raise serializers.ValidationError("Zone not found.")