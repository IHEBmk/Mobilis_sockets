import json
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from server.authentication import CustomJWTAuthentication
from server.models import Wilaya
from rest_framework.permissions import IsAuthenticated


class Wilaya_to_supabase(APIView):
    authentication_classes = []
    permission_classes = []   
    def post(self, request):
        csv = request.FILES.get("file")
        wilaya_attr = request.data.get("wilaya_attr")

        if not csv:
            return Response({'error': 'File is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not wilaya_attr:
            return Response({'error': 'wilaya_attr is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            df = pd.read_csv(csv)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        wilayas_to_create = []
        errors = []
        for index, row in df.iterrows():
            name = row.get(wilaya_attr)
            if not name:
                errors.append({'row': index + 1, 'errors': 'Missing wilaya name'})
                continue

            wilayas_to_create.append(
                Wilaya(id=uuid.uuid4(), name=name)
            )

        try:
            Wilaya.objects.bulk_create(wilayas_to_create)
        except Exception as e:
            return Response({'error': f'Failed to insert batch: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if errors:
            return Response({'message': 'Imported with some errors', 'errors': errors}, status=status.HTTP_207_MULTI_STATUS)

        return Response({'message': 'CSV imported successfully'}, status=status.HTTP_201_CREATED)






class GetWilayas(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        wilayas=Wilaya.objects.all()
        wilayas_list=[]
        for wilaya in wilayas:
            wilayas_list.append({'id':wilaya.id,'name':wilaya.name})
        return Response({'wilayas': wilayas_list}, status=status.HTTP_200_OK)
    


class GetGeojson(APIView):
    def get(self, request):
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role=='manager':
            geojson=user.wilaya.geojson
            if geojson:
                return Response({'geojson':json.loads(geojson)}, status=status.HTTP_200_OK, content_type='application/json')
            else:
                return Response({'error': 'geojson not found'}, status=status.HTTP_404_NOT_FOUND)
        elif user.role=='admin':
            wilayas=Wilaya.objects.all()
            geojson={}
            for wilaya in wilayas:
                geojson[str(wilaya.name)]=json.loads(wilaya.geojson) if wilaya.geojson else None
            if geojson:
                return Response({'geojson':(geojson)}, status=status.HTTP_200_OK, content_type='application/json')
            else:
                return Response({'error': 'geojson not found'}, status=status.HTTP_404_NOT_FOUND)
        elif user.role=='agent':
            return Response({'error': 'You are not authorized to view this page'}, status=status.HTTP_403_FORBIDDEN)
