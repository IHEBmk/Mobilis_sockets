from datetime import datetime
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from server.authentication import CustomJWTAuthentication
from server.models import Commune, Coordinates, PointOfSale, User, Wilaya, Zone
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Coordinates, User
from .authentication import CustomJWTAuthentication
from rest_framework.permissions import IsAuthenticated

class GetCoordinates(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        if user.role == 'agent':
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)

        if user.role == 'manager':
            users = User.objects.filter(manager=user)
            coordinates = Coordinates.objects.filter(user__in=users).select_related('user')
        elif user.role == 'admin':
            coordinates = Coordinates.objects.select_related('user').all()
        else:
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)

        serialized_coordinates = []
        for coord in coordinates:
            serialized_coordinates.append({
                'first_name': coord.user.first_name,
                'last_name': coord.user.last_name,
                'latitude': coord.lattitude,
                'longitude': coord.longitude,
            })

        return Response({'coordinates': serialized_coordinates}, status=status.HTTP_200_OK)


    
    
    
    
    
class RefreshCoordinates(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user = request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'agent':
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        id=uuid.uuid4()
        created_at=datetime.now()
        try:
            coordinate=Coordinates.objects.get(user=user.id)
            coordinate.latitude=latitude
            coordinate.longitude=longitude
            coordinate.save()
            return Response({'message':'coordinates updated successfully'}, status=status.HTTP_200_OK)
        except Coordinates.DoesNotExist:
            new_coordinate=Coordinates(id=id,user=user,lattitude=latitude,longitude=longitude,created_at=created_at)
            new_coordinate.save()
            return Response({'message':'coordinates added successfully'}, status=status.HTTP_200_OK)
        
            