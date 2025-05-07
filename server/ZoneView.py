import json
from datetime import datetime
import random
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import pandas as pd
from server.authentication import CustomJWTAuthentication

from server.ZoneSerializer import ZoneSerializer
from server.models import Commune, PointOfSale, User, Wilaya
from rest_framework.permissions import IsAuthenticated
from server.zoning import assign_communes_from_geojson, create_balanced_zones, export_zones_to_geojson, generate_zone_boundaries, load_data





class GenerateZones(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user = request.user
        wilaya_id = request.data.get('wilaya_id')
        number_of_zones = int(request.data.get('number_of_zones'))
        csv_file = request.FILES.get('csv_file')
        lat_col = request.data.get('lat_col')
        lon_col = request.data.get('lon_col')
        balance_type = request.data.get('balance_type')
        
        # Validate required parameters
        if not csv_file:
            return Response({'error': 'CSV file is required'}, status=status.HTTP_400_BAD_REQUEST)
            
        if not id:
            return Response({'error': 'User ID is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # try:
        #     user = User.objects.get(id=id)
        # except User.DoesNotExist:
        #     return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        
        if user.role == 'admin':
                wilaya_id = request.data.get('wilaya_id')
                if not wilaya_id:
                    return Response({'error': 'Wilaya ID is required for admin users'}, status=status.HTTP_400_BAD_REQUEST)
        elif user.role == 'manager':
                if not user.wilaya:
                    return Response({'error': 'Manager does not have an assigned wilaya'}, status=status.HTTP_400_BAD_REQUEST)
                wilaya_id = str(user.wilaya.id)
        else:
                return Response({'error': 'You do not have permission to generate zones'}, status=status.HTTP_403_FORBIDDEN)
        
        # Get the wilaya
        try:
            wilaya = Wilaya.objects.get(id=wilaya_id)
        except Wilaya.DoesNotExist:
            return Response({'error': 'Wilaya not found'}, status=status.HTTP_400_BAD_REQUEST)
        
        # # Check if zones are already generated
        # if wilaya.geojson is not None:
        #     return Response({'error': 'Zones already generated'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get the communes for this wilaya
        communes = list(Commune.objects.filter(wilaya=wilaya_id).values())
        commune_ids = [commune['id'] for commune in communes]
        
        # Get points of sale in this wilaya
        pdvs_to_zone = list(PointOfSale.objects.filter(commune__in=commune_ids).values())
        
        if not pdvs_to_zone:
            return Response({'error': 'No points of sale found for this wilaya'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Set balance coefficients based on balance_type
        if not lat_col:
            lat_col = 'Latitude'
        if not lon_col:
            lon_col = 'Longitude'
        if not balance_type:
            balance_type = 'balanced'
        if balance_type == "points":
            points_coef = 10
            distance_coef = 0.1
        elif balance_type == "distance":
            points_coef = 0.1
            distance_coef = 10
        else:
            points_coef = 1
            distance_coef = 1
        
        # Load data from CSV and assign communes using GeoJSON if needed
        df = load_data(csv_file, lat_col, lon_col)
        
        # Assign communes using GeoJSON if needed
        if not ('Commune' in df.columns and df['Commune'].notna().all()):
            df = assign_communes_from_geojson(df, 
                "https://qlluxlhcvjnlicxzxwry.supabase.co/storage/v1/object/sign/communes/geoBoundaries-DZA-ADM3.geojson?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJjb21tdW5lcy9nZW9Cb3VuZGFyaWVzLURaQS1BRE0zLmdlb2pzb24iLCJpYXQiOjE3NDQ0NzYzOTMsImV4cCI6MzMyODA0NzYzOTN9.zEbNpeH6f2gp-n3Jrue20cC3cHAq4wZ00Duy6IeEsGs", 
                lat_col, lon_col)
        
        # Create balanced zones
        df,zones, zone_workloads, zone_communes = create_balanced_zones(
            df, number_of_zones, lat_col, lon_col, 
            points_coef=points_coef, distance_coef=distance_coef
        )
        
        # Generate zone boundaries
        zone_polygons = generate_zone_boundaries(zones)
        
        # Get available managers for assignment
        available_managers = User.objects.filter(role='manager', wilaya=wilaya_id)
        manager_count = available_managers.count()
        
        if manager_count < number_of_zones:
            # Not enough managers, we'll reuse some
            managers_list = list(available_managers)
            manager_assignments = random.choices(managers_list, k=number_of_zones)
        else:
            # We have enough managers, randomly select without replacement
            managers_list = list(available_managers)
            manager_assignments = random.sample(managers_list, number_of_zones)
        
        # Map zone_id to manager
        zone_managers = dict(zip([str(i) for i in zones.keys()], manager_assignments))
        
        errors = []
        created_zones = {}
        
        # Create zones and assign managers
        for zone_id, zone_df in zones.items():
            try:
                zone_id_str = str(zone_id)
                manager = zone_managers.get(zone_id_str)
                
                # Format zone name
                name = f"{wilaya.name}_{zone_id}"
                
                # Create new zone
                zone_uuid = uuid.uuid4()
                created_at = datetime.now()
                
                new_zone_dict = {
                    'id': zone_uuid,
                    'created_at': created_at,
                    'commune': None,  # Will be updated later if needed
                    'name': name,
                    'manager': manager.id if manager else None
                }
                
                new_zone = ZoneSerializer(data=new_zone_dict)
                if new_zone.is_valid():
                    zone = new_zone.save()
                    created_zones[zone_id] = zone
                else:
                    errors.append({'zone_id': zone_id, 'errors': new_zone.errors})
                    continue
                
                # Update all PDVs in this zone with the zone_id
                for _, row in zone_df.iterrows():
                    try:
                        # Find matching PDV by coordinates
                        pdv = PointOfSale.objects.filter(
                            latitude=row[lat_col],
                            longitude=row[lon_col]
                        ).first()
                        
                        if pdv:
                            pdv.zone = zone
                            pdv.save()
                    except Exception as e:
                        errors.append({
                            'type': 'pdv_update_error',
                            'zone_id': zone_id,
                            'latitude': row[lat_col],
                            'longitude': row[lon_col],
                            'error': str(e)
                        })
                
            except Exception as e:
                errors.append({
                    'type': 'zone_creation_error',
                    'zone_id': zone_id,
                    'error': str(e)
                })
        
        # Export zones to GeoJSON and update wilaya
        try:
            # Generate GeoJSON directly as a dictionary (no file output)
            geojson_data = export_zones_to_geojson(
                df, zones, zone_workloads, zone_communes, zone_polygons,
                output_path=None,  # Don't write to file
                commune_geojson="https://qlluxlhcvjnlicxzxwry.supabase.co/storage/v1/object/sign/communes/geoBoundaries-DZA-ADM3.geojson?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1cmwiOiJjb21tdW5lcy9nZW9Cb3VuZGFyaWVzLURaQS1BRE0zLmdlb2pzb24iLCJpYXQiOjE3NDQ0NzYzOTMsImV4cCI6MzMyODA0NzYzOTN9.zEbNpeH6f2gp-n3Jrue20cC3cHAq4wZ00Duy6IeEsGs"
            )
            
            # Convert the dict to a JSON string and save it to the wilaya object
            if isinstance(geojson_data, dict):
                wilaya.geojson = json.dumps(geojson_data)
                wilaya.save()
                print(f"Successfully saved GeoJSON to wilaya {wilaya.name}")
            else:
                errors.append({
                    'type': 'geojson_generation_error',
                    'error': "Expected dict result from export_zones_to_geojson"
                })
        except Exception as e:
            errors.append({
                'type': 'geojson_save_error',
                'error': str(e)
            })
        
        response_data = {
            'success': True,
            'zones_created': len(created_zones),
            'errors': errors if errors else None
        }
        
        if errors:
            response_data['warning'] = f"Completed with {len(errors)} errors"
        
        return Response(response_data, status=status.HTTP_200_OK)
        
                
        
        


class GetGeojsonWilaya(APIView):
    def post(self, request):
        id = request.data.get('id')
        wilaya_id=request.data.get('wilaya_id')
        if not id or not wilaya_id:
            return Response({'error': 'ID is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(id=id)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='manager' and user.role!='admin':
            return Response({'error': 'you can\'t retrieve zones'}, status=status.HTTP_401_UNAUTHORIZED)
        if user.role=='manager':
            if wilaya_id!=str(user.wilaya.id):
                return Response({'error': 'you can\'t generate zones'}, status=status.HTTP_400_BAD_REQUEST)
        geosjon_txt=user.wilaya.geojson
        geosjon=json.loads(geosjon_txt)
        return Response({'message': 'Geojson generated successfully',
                         'geojson': geosjon}, status=status.HTTP_200_OK)



        




