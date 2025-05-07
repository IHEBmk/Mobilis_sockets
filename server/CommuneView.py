

from io import StringIO
import uuid
import pandas as pd
import requests
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from server.CommuneSerializer import CommuneSerializer
from server.authentication import CustomJWTAuthentication
from rest_framework.permissions import IsAuthenticated
from server.models import Commune, Wilaya

class Commune_to_supabase(APIView):
    # authentication_classes = [CustomJWTAuthentication]
    # permission_classes = [IsAuthenticated]
    def post(self, request):
        csv = request.FILES.get("file")
        commune_attr = request.data.get("commune_attr")
        wilaya_attr = request.data.get("wilaya_attr")

        if not commune_attr:
            return Response({'error': 'commune_attr is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not wilaya_attr:
            return Response({'error': 'wilaya_attr is required'}, status=status.HTTP_400_BAD_REQUEST)
        if not csv:
            return Response({'error': 'File is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            df = pd.read_csv(csv)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        wilaya_map = {w["name"]: w["id"] for w in Wilaya.objects.all().values()}
        communes_to_create = []
        errors = []

        for index, row in df.iterrows():
            row_dict = row.to_dict()
            wilaya_id = wilaya_map.get(row_dict.get(wilaya_attr))

            if not wilaya_id:
                errors.append({'row': index + 1, 'errors': 'Wilaya'+ str(row_dict.get(wilaya_attr)) + ' not found'})
                continue

            commune = Commune(
                id=uuid.uuid4(),
                name=row_dict.get(commune_attr),
                wilaya_id=wilaya_id
            )
            communes_to_create.append(commune)

        try:
            Commune.objects.bulk_create(communes_to_create)
        except Exception as e:
            return Response({'error': f'Failed to insert batch: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if errors:
            return Response({'message': 'Imported with some errors', 'errors': errors}, status=status.HTTP_207_MULTI_STATUS)

        return Response({'message': 'CSV imported successfully'}, status=status.HTTP_201_CREATED)





class GetCommunes(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        communes=Commune.objects.all()
        communes_list=[]
        for commune in communes:
            communes_list.append({'id':commune.id,'name':commune.name,'wilaya':commune.wilaya.id})
        return Response({'communes':communes_list},status=status.HTTP_200_OK)