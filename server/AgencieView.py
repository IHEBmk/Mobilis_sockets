from datetime import datetime
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import pandas as pd
from server.Visit_paling_model import plan_multiple_days
from server.authentication import CustomJWTAuthentication
from server.models import  Agence, PointOfSale, User, Visit, Wilaya
from rest_framework.permissions import IsAuthenticated









class UploadAgencies(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        csv = request.FILES.get("file")
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if not csv:
            return Response({'error': 'File is required'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='admin':
            return Response({'error': 'you can\'t upload agencies'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        wilaya_map = {w["name"]: w["id"] for w in Wilaya.objects.all().values()}
        Agencies_to_create = []
        errors = []
        maxl=0
        maxll=0
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            wilaya_name = str(row_dict.get('Wilaya')).strip().title()
            wilaya_id = wilaya_map.get(wilaya_name)
            if not wilaya_id:
                errors.append({'row': index + 1, 'errors': 'Wilaya '+ str(row_dict.get('Wilaya').capitalize()) + ' not found'})
                continue
            name=row_dict.get('agence') if row_dict.get('agence')!=None else 'Agence'
            latitude=row_dict.get('coord_y') if row_dict.get('coord_y')!=None else 0
            longitude=row_dict.get('coord_x') if row_dict.get('coord_x')!=None else 0
            temp={'id':uuid.uuid4(),'name':name,'wilaya':(Wilaya.objects.get(id=wilaya_id)),'latitude':latitude,'longitude':longitude}
            Agence_ = Agence(**temp)
            Agencies_to_create.append(Agence_)

        try:

            Agence.objects.bulk_create(Agencies_to_create)
        except Exception as e:
            return Response({'error': f'Failed to insert batch: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if errors:
            return Response({'message': 'Imported with some errors', 'errors': errors}, status=status.HTTP_207_MULTI_STATUS)

        return Response({'message': 'CSV imported successfully'}, status=status.HTTP_201_CREATED)





class AddAgence(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        user=request.user
        data=request.data
        if not data:
            return Response({'error': 'data is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='admin': 
            return Response({'error': 'you can\'t add agence'}, status=status.HTTP_400_BAD_REQUEST)
        if not data.get('name'):
            return Response({'error': 'name is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not data.get('longitude'):
            return Response({'error': 'longitude is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not data.get('latitude'):
            return Response({'error': 'latitude is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not data.get('wilaya'):
            return Response({'error': 'wilaya is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            wilaya=Wilaya.objects.get(name=data.get('wilaya'))
        except Wilaya.DoesNotExist:
            return Response({'error': 'Wilaya not found'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            existing_agence=Agence.objects.get(latitude=data.get('latitude'),longitude=data.get('longitude'))
            return Response({'error': 'Agence already exists'}, status=status.HTTP_400_BAD_REQUEST)
        except Agence.DoesNotExist:
            pass
        Agence.objects.create(id=uuid.uuid4(),name=data.get('name'),latitude=data.get('latitude'),longitude=data.get('longitude'),wilaya=wilaya)
        return Response({'message': 'Agence added successfully'}, status=status.HTTP_201_CREATED)
    
    
