import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from server.authentication import CustomJWTAuthentication
from server.models import Commune, PointOfSale, User, Wilaya, Zone
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone

class Pdv_To_supabase(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user=request.user
        wilaya=request.data.get('wilaya')
        if not wilaya:
            return Response({'error':'wilaya is missing'}, status=status.HTTP_400_BAD_REQUEST)
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='admin' and user.role!='manager':
            return Response({'error': 'you can\'t insert pdv'}, status=status.HTTP_400_BAD_REQUEST)
        csv_file = request.FILES.get("file")
        if not csv_file:
            return Response({'error':"No file found"},status=status.HTTP_404_NOT_FOUND)

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        communes=list(Commune.objects.all().values())
        rows_to_save = []
        errors=[]
        for index, row in df.iterrows():
            row_dict = row.to_dict()
            commune_id = next((item["id"] for item in communes if item["name"] == row_dict['Commune']), None)
            if not commune_id:
                errors.append({'errors':str(row_dict['Commune'])+' not found'})
                # commune_id=uuid.uuid4()
                # new_commune=Commune(id=commune_id,name=row_dict['Commune'],wilaya_id=wilaya)
                # new_commune.save()
                # communes=list(Commune.objects.all().values())
                continue
            pos=PointOfSale(
                id=uuid.uuid4(),
                latitude=row_dict['Latitude'],
                longitude=row_dict['Longitude'],
                name=row_dict['Nom du point de vente'],
                commune_id=commune_id,
                status=1,
                created_at=timezone.now()
            )
            rows_to_save.append(pos)
            
        PointOfSale.objects.bulk_create(rows_to_save)
        if errors:
            return Response({'error': errors}, status=status.HTTP_400_BAD_REQUEST)
        total=len(rows_to_save)
        return Response({'message': 'CSV imported successfully and inserted '+str(total)+" rows"}, status=status.HTTP_201_CREATED)







# class AddPdv(APIView):
#     authentication_classes = [CustomJWTAuthentication]
#     permission_classes = [IsAuthenticated]
#     def post(self,request):
#         data=request.data
#         user=request.user
#         if not user:
#             return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
#         latitude=data['latitude']
#         longitude=data['longitude']
#         commune=data['commune']
#         new_pdv={}
#         new_pdv['latitude']=latitude
#         new_pdv['longitude']=longitude
#         new_pdv['commune']=commune
#         new_pdv['id']=uuid.uuid4()
#         commune=Commune.objects.get(id=commune)
#         if not latitude or not longitude or not commune:
#             return Response({'error': 'ID is required'}, status=status.HTTP_400_BAD_REQUEST)
        
#         existing_pdv = PointOfSale.objects.filter(latitude=latitude, longitude=longitude)
#         if existing_pdv.exists():
#             #existing_pdv.delete()
#             return Response({'error': 'Point of sale already exists  '}, status=status.HTTP_400_BAD_REQUEST)
        
#         if user.role!='admin' and user.role!='manager':
#             return Response({'error': 'you can\'t insert pdv'}, status=status.HTTP_400_BAD_REQUEST)
#         if user.role=='manager':
#             if commune.wilaya.id!=user.wilaya.id:
#                 return Response({'error': 'you can\'t insert pdv'}, status=status.HTTP_400_BAD_REQUEST)
            
#         serializer=PointOfSaleSerializer(data=new_pdv)
#         if serializer.is_valid():
#             serializer.save()
            
#             if commune.wilaya.geojson==None:
#                 return Response({'message: insertedsuccessfully'},status=status.HTTP_201_CREATED)
#             else:
#                 json_data=commune.wilaya.geojson
#                 pdvs=list(PointOfSale.objects.filter(commune=commune.id).values())
#                 df=pd.DataFrame(pdvs)
#                 number_of_zones=df['zone_id'].unique().shape[0]
#                 print(number_of_zones)
#                 print(df['zone_id'].head())
#                 print(df.columns)
#                 zone_ids=[]
#                 for pdv in pdvs:
#                     temp=PointOfSale.objects.get(id=pdv['id'])
#                     if temp.zone==None:
#                         continue
#                     temp.zone=None
#                     temp.save()
#                     zone_ids.append(pdv['zone_id'])
#                     pdv['zone_id']=None
#                 zone_ids=set(zone_ids)
#                 for id in zone_ids:
#                     zone=Zone.objects.get(id=id)
#                     zone.delete()
#                 print(df['zone_id'].head())
#                 df,gdf = cluster_communes(df,number_of_zones+3)
#                 errors=[]
#                 zones={}
#                 print("clustering")
#                 print(df[['zone_id','Cluster']].head())
#                 for zone in df['Cluster'].unique():
#                     id =uuid.uuid4()
#                     created_at=datetime.now()
#                     new_zone_dict={'id':id,'created_at':created_at,'commune':commune.id,'zone':zone}
#                     new_zone=ZoneSerializer(data=new_zone_dict)
#                     if new_zone.is_valid():
#                         df.loc[df['Cluster'] == zone, 'zone_id'] = id
#                         new_zone.save()
#                         zones[id]=new_zone.instance
#                     else:
#                         print('error')
#                         errors.append({'errors': new_zone.errors})
#                 print(df[['zone_id','Cluster']].head())

#                 point_updates = []
#                 for _, row in df.iterrows():
#                     pos = PointOfSale.objects.get(id=row['id'])
#                     pos.zone = zones[row['zone_id']]
#                     point_updates.append(pos)
#                 with transaction.atomic():
#                     PointOfSale.objects.bulk_update(point_updates, ['zone'])
#                 new_geojson=update_commune_geojson(df,commune.id,json_data)
#                 commune.wilaya.geojson=json.dumps(new_geojson)
#                 commune.wilaya.save()
#                 return Response({'message': 'inserted successfully',
#                                  'geojson':new_geojson },status=status.HTTP_201_CREATED)
            
            
#         else:
#             print(serializer.errors)
#             return Response({'message': 'failed to insert',
#                              'errors':serializer.errors},status=status.HTTP_400_BAD_REQUEST)
    
    
    




from django.db.models import Prefetch

class GetPdv(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, format=None):
        user = request.user
        if not user:
            return Response({'message': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)

        if user.role not in ['admin', 'manager', 'agent']:
            return Response({'message': 'Invalid method'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

        if user.role == 'admin':
            pdvs = PointOfSale.objects.select_related('commune', 'zone', 'manager')
        elif user.role == 'manager':
            communes = Commune.objects.filter(wilaya=user.wilaya)
            pdvs = PointOfSale.objects.filter(commune__in=communes).select_related('commune', 'zone', 'manager')
        elif user.role == 'agent':
            pdvs = PointOfSale.objects.filter(manager=user).select_related('commune', 'zone', 'manager')

        result = []
        for pdv in pdvs:
            pdv_data = {
                'id': pdv.id,
                'name': pdv.name,
                'latitude': pdv.latitude,
                'longitude': pdv.longitude,
                'status': pdv.status,
                'created_at': pdv.created_at,
                'commune': pdv.commune.name if pdv.commune else None,
                'zone': pdv.zone.name if pdv.zone else None,
                'manager': f'{pdv.manager.first_name} {pdv.manager.last_name}' if pdv.manager else None,
            }
            result.append(pdv_data)

        return Response({'pdvs': result}, status=status.HTTP_200_OK)

        
    
    
    
    
    
    
class UpdateStatusPdv(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        data=request.data
        pdv=data.get('pdv')
        status_=data.get('status')
        if  not data.get('pdv'):
            return Response({'error':'id is required'}, status=status.HTTP_400_BAD_REQUEST)
        user=request.user
        
        pdv=PointOfSale.objects.get(id=data.get('pdv'))
        commune=Commune.objects.get(id=pdv.commune.id)
        if not user or not pdv:
            return Response({'error':'unauthorized'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role=='manager':
            wilaya=Wilaya.objects.get(id=commune.wilaya)
            if user.wilaya!=wilaya:
                return Response({'error':'you can\'t update this pdv'}, status=status.HTTP_400_BAD_REQUEST)
            pdv.status=status_
            pdv.save()
            return Response({'message':"status updated successfully"}, status=status.HTTP_200_OK)
        elif user.role=='admin':
            pdv.status=status_
            pdv.save()
            return Response({'message':"status updated successfully"}, status=status.HTTP_200_OK)
        else:
            return Response({'error':'you can\'t update this pdv, unauthorized'}, status=status.HTTP_400_BAD_REQUEST)
            
            
            
            
            
            

    





class DeletePdv(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        data=request.data
        pdv=data.get('pdv')
        status=data.get('status')
        if  not data.get('pdv'):
            return Response({'error':'id is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user=request.user
            if not user:
                return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
            pdv=PointOfSale.objects.get(id=data.get('pdv'))
        except User.DoesNotExist or PointOfSale.DoesNotExist or Commune.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        commune=Commune.objects.get(id=pdv.commune)
        if user.role=='manager':
            wilaya=Wilaya.objects.get(id=commune.wilaya)
            if user.wilaya!=wilaya:
                return Response({'error':'you can\'t update this pdv'}, status=status.HTTP_400_BAD_REQUEST)
            pdv.delete()
            return Response({'message':"pdv deleted successfully"}, status=status.HTTP_200_OK)
        elif user.role=='admin':
            pdv.delete()
            return Response({'message':"deleted successfully"}, status=status.HTTP_200_OK)
        else:
            return Response({'error':'you can\'t delete this pdv, unauthorized'}, status=status.HTTP_400_BAD_REQUEST)
        