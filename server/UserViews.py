
import hashlib
import uuid
from django.forms import model_to_dict
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from server.models import User, Wilaya
from .Userserializers import UserAssignSerializer, UserLoginSerializer, UserSignupSerializer
from rest_framework.permissions import IsAuthenticated
from .authentication import CustomJWTAuthentication, get_tokens_for_user
from django.contrib.auth.hashers import make_password
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
import hashlib
import uuid
from .models import User
from .authentication import CustomJWTAuthentication


class SignupView(APIView):
    authentication_classes = []
    permission_classes = []     
    
    def post(self, request):
        print(request.data)
        serializer = UserSignupSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            
            # Generate JWT tokens
            tokens = get_tokens_for_user(user)
            
            return Response({
                'message': 'User created successfully!',
                'tokens': tokens,
                'uuid':user.id
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class GetUsers(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user = request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role == 'admin':
            users = User.objects.all()
        elif user.role == 'manager':
            users = User.objects.filter(manager=user)
        elif user.role == 'agent':
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)
        else:
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)
        serialized_users = []
        for user in users:
            serialized_users.append(model_to_dict(user))
        return Response({'users': serialized_users}, status=status.HTTP_200_OK)
class LoginView(APIView):
    authentication_classes = []
    permission_classes = []
    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            
            # Generate JWT tokens
            tokens = get_tokens_for_user(user)
            
            return Response({
                'message': 'Login successful',
                'user_id': str(user.id),
                'wilaya_name': user.wilaya.name if user.wilaya else None,
                'wilaya_id': str(user.wilaya.id) if user.wilaya else None,
                
                
                'role': user.role,
                'tokens': tokens
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AssignZoneView(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        # Add the authenticated user as the assigner if not provided
        data = request.data.copy()
        if 'assigner' not in data:
            data['assigner'] = str(request.user.id)
            
        serializer = UserAssignSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response({'message': 'Zone assigned successfully!'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    
    
    
    
class GenerateWilayasManagers(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user=request.user
        data=request.data
        if ['password','email']!=list(data.keys()):
            return Response({'error': 'missing data'}, status=status.HTTP_400_BAD_REQUEST)
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='admin':
            return Response({'error': 'you can\'t generate managers'}, status=status.HTTP_400_BAD_REQUEST)
        wilayas=(Wilaya.objects.all())
        i=1
        users=[]
        for wilaya in wilayas:
            new_user=User(id=uuid.uuid4(),first_name='manager',last_name=str(i),password= hashlib.sha256(str(data['password']+str(i)).encode()).hexdigest(),email=data['email'],role='manager',wilaya=wilaya,phone='123456789')
            i+=1
            users.append(new_user)
        User.objects.bulk_create(users)
        return Response({
    'message': 'Managers generated successfully',
    'managers': [model_to_dict(user) for user in users]
}, status=status.HTTP_201_CREATED)
    





class GenerateManager(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user=request.user
        data=request.data
        if ['password','email','wilaya']!=list(data.keys()):
            return Response({'error': 'missing data'}, status=status.HTTP_400_BAD_REQUEST)
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='admin' and user.role!='manager':
            return Response({'error': 'you can\'t generate managers'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            wilaya=Wilaya.objects.get(id=data['wilaya'])
        except Wilaya.DoesNotExist:
            return Response({'error': 'Wilaya not found'}, status=status.HTTP_400_BAD_REQUEST)
        new_user=User(id=uuid.uuid4(),first_name='manager',last_name='',password= hashlib.sha256(str(data['password']).encode()).hexdigest(),email=data['email'],role='manager',wilaya=wilaya,phone='123456789')
        new_user.save()
        return Response({'message': 'Manager generated successfully','manager':model_to_dict(new_user)}, status=status.HTTP_201_CREATED)
        
        
        


class GenerateAgents(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]    
    
    def post(self, request):
        user = request.user
        data = request.data

        if set(data.keys()) != {'password', 'email', 'num'}:
            return Response({'error': 'missing data'}, status=status.HTTP_400_BAD_REQUEST)
        
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)

        if user.role != 'manager':
            return Response({'error': 'You can\'t generate agents'}, status=status.HTTP_403_FORBIDDEN)

        new_users = []
        agent_credentials = []

        for i in range(int(data['num'])):
            unique_email = f"{data['email'].split('@')[0]}+{i}@{data['email'].split('@')[1]}"
            raw_password = f"{data['password']}{i}"
            hashed_password = hashlib.sha256(raw_password.encode()).hexdigest()
            new_user = User(
                id=uuid.uuid4(),
                first_name='agent',
                last_name=str(i),
                password=hashed_password,
                email=unique_email,
                role='agent',
                wilaya=user.wilaya,
                phone='123456789',
                manager=user
            )
            new_users.append(new_user)
            temp=model_to_dict(new_user)
            temp['password']=raw_password
            agent_credentials.append(temp)

        User.objects.bulk_create(new_users)

        return Response({
            'message': 'Agents generated successfully',
            'agents': agent_credentials
        }, status=status.HTTP_201_CREATED)








class GenerateWilayaManagers(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user=request.user
        data=request.data
        if ['password','email','num']!=list(data.keys()):
            return Response({'error': 'missing data'}, status=status.HTTP_400_BAD_REQUEST)
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='manager':
            return Response({'error': 'you can\'t generate managers'}, status=status.HTTP_400_BAD_REQUEST)
        new_users = []
        manager_credentials = []
        for i in range(int(data['num'])):
            unique_email = f"{data['email'].split('@')[0]}+{i}@{data['email'].split('@')[1]}"
            raw_password = f"{data['password']}{i}"
            hashed_password = hashlib.sha256(raw_password.encode()).hexdigest()
            new_user = User(
                id=uuid.uuid4(),
                first_name='manager',
                last_name=str(i),
                password=hashed_password,
                email=unique_email,
                role='manager',
                wilaya=user.wilaya,
                phone='123456789'
            )
            new_users.append(new_user)
            temp=model_to_dict(new_user)
            temp['password']=raw_password
            manager_credentials.append(temp)
        User.objects.bulk_create(new_users)
        return Response({
            'message': 'Managers generated successfully',
            'managers': manager_credentials
        }, status=status.HTTP_201_CREATED)
        
        
        
        
        
        
        
        
        
        
class GetUsers(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user = request.user

        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)

        if user.role == 'admin':
            users = User.objects.all()
        elif user.role == 'manager':
            users = User.objects.filter(manager=user)
        elif user.role == 'agent':
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)
        else:
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)

        serialized_users = []

        for user in users:
            serialized_users.append(model_to_dict(user))
        return Response({'users': serialized_users}, status=status.HTTP_200_OK)