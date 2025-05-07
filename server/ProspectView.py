from datetime import datetime
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import pandas as pd
from server.authentication import CustomJWTAuthentication
from server.models import  Commune, PointOfSale, Prospect, User, Visit
from rest_framework.permissions import IsAuthenticated



class AddProspect(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        data=request.data
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='agent':
            return Response({'error': 'you can\'t add a prospect'}, status=status.HTTP_400_BAD_REQUEST)
        if ['name','phone','address','latitude','longitude','commune','registre_de_commerce','numero_fiscal']!=list(data.keys()):
            return Response({'error': 'missing data'}, status=status.HTTP_400_BAD_REQUEST)
        if user.wilaya!=Commune.objects.get(id=data['commune']).wilaya:
            return Response({'error': 'you can\'t add a prospect in this commune'}, status=status.HTTP_400_BAD_REQUEST)
        new_prospect=Prospect(
            id=uuid.uuid4(),
            pos_name=data['name'],
            longitude=data['longitude'],
            latitude=data['latitude'],
            status='pending',
            phone_number=data['phone'],
            street_address=data['address'],
            registre_de_commerce=data['registre_de_commerce'],
            numero_fiscal=data['numero_fiscal'],
            commune=Commune.objects.get(id=data['commune']).id
        )
        new_prospect.save()
        return Response({'message': 'Prospect added successfully'}, status=status.HTTP_201_CREATED)