import base64
from collections import defaultdict
from datetime import datetime, timedelta
from django.utils import timezone
import uuid
from django.forms import model_to_dict
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
import pandas as pd
from server.Visit_paling_model import plan_multiple_days
from server.authentication import CustomJWTAuthentication
from server.models import  PointOfSale, User, Visit
from rest_framework.permissions import IsAuthenticated



from datetime import datetime, timedelta
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.timezone import make_aware



class GetVisits(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role=='admin':
            visits=Visit.objects.all()
        elif user.role=='manager':
            supervised_agents=User.objects.filter(manager=user.id).values()
            visits=Visit.objects.filter(agent__in=supervised_agents)
        elif user.role=='agent':
            visits=Visit.objects.filter(agent=user.id,status='pending')
        else:
            return Response({'error': 'you can\'t retrieve visits'}, status=status.HTTP_401_UNAUTHORIZED)
        visits_list=[]
        for visit in visits:
            pdv=visit.pdv
            visits_list.append({'id':visit.id,'pdv':pdv.name,'deadline':visit.deadline})
        return Response({'visits': visits_list}, status=status.HTTP_200_OK)
    
    
    
class VisitPdv(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        remake = data.get('remake')
        speed_kmph = data.get('speed_kmph', 60)

        if not data.get('pdv'):
            return Response({'error': 'Missing pdv or id'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = request.user
            if not user:
                return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)

            pdv = PointOfSale.objects.get(id=data.get('pdv'))
        except (User.DoesNotExist, PointOfSale.DoesNotExist):
            return Response({'error': 'User or Point of Sale not found'}, status=status.HTTP_400_BAD_REQUEST)

        if user.role != "agent":
            return Response({'error': "You are not an agent"}, status=status.HTTP_403_FORBIDDEN)

        try:
            # Get the visit that is scheduled for the Point of Sale and agent
            visit = Visit.objects.get(pdv=pdv, agent=user, status='scheduled')

            # Check if the visit deadline is tomorrow or later
            if visit.deadline >= timezone.now():
                # Get today's date in the local timezone
                today = timezone.localdate()

                # Filter visits that are scheduled for today and not yet completed
                today_visits = Visit.objects.filter(agent=user, validated=1, status='scheduled', deadline__date=today)

                # If there are any scheduled visits for today, return an error
                if today_visits.exists():
                    return Response({'error': 'You can\'t visit non-scheduled visits until you finish the current scheduled visits'}, 
                                    status=status.HTTP_400_BAD_REQUEST)
            visit.status = 'visited'
            visit.visit_time=timezone.now()
            visit.save()

            pdv.last_visit = timezone.now()
            pdv.save()

            if visit.deadline <= timezone.now():
                return Response({'message': "Visited successfully"}, status=status.HTTP_200_OK)

            if remake:
                now = timezone.now()
                visits = Visit.objects.filter(agent=user, validated=1, status='scheduled', deadline__gte=now)
                data_points = [
                    {'id': visit.pdv.id, 'name': visit.pdv.name, 'longitude': visit.pdv.longitude, 'latitude': visit.pdv.latitude}
                    for visit in visits
                ]
                start_point = {
                    'name': "Start",
                    'longitude': user.Agence.longitude,
                    'latitude': user.Agence.latitude
                }

                deadline = now
                points=[]
                for visit in visits:
                    points.append(visit.pdv.id)
                    visit.delete()
                deadline=user.deadline
                future_start = now + timedelta(days=1)
                number_of_days = (deadline - future_start).days
                daily_limit_minutes = 7 * 60
                total_time_minutes = number_of_days * daily_limit_minutes
                # Assuming you are planning again with all PDVs managed by user
                data_points = list(PointOfSale.objects.filter(manager=user,id__in=points).values())
                routes, edges, estimates = plan_multiple_days([start_point] + data_points, daily_limit_minutes, speed_kmph)
                new_visits = []
                for day_index, route in enumerate(routes):
                    for order, point_name in enumerate(route[1:], start=1):  # skip start point
                        pos = next((p for p in data_points if p['name'] == point_name), None)
                        if not pos:
                            continue
                        new_visit = Visit(
                            id=uuid.uuid4(),
                            deadline=future_start + timedelta(days=day_index),
                            agent=user,
                            pdv_id=pos['id'],
                            status='scheduled',
                            order=order,
                            validated=1
                        )
                        new_visits.append(new_visit)

                Visit.objects.bulk_create(new_visits)
                return Response({
                    'message': 'Visits rescheduled successfully',
                    
                }, status=status.HTTP_201_CREATED)

        except Visit.DoesNotExist:
            return Response({'error': 'Visit was not scheduled'}, status=status.HTTP_400_BAD_REQUEST)
        
    
    
    
    
    
    
    
    
class MakePlanning(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        user = request.user
        speed_kmph = float(data.get('speed_kmph', 60))
        
        remake = data.get('remake')
        cvi_id = data.get('cvi')
        deadline_str = data.get('deadline')

        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'manager':
            return Response({'error': 'You can\'t make planning'}, status=status.HTTP_400_BAD_REQUEST)
        if not deadline_str or not cvi_id:
            return Response({'error': 'Deadline and CVI are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            deadline = datetime.strptime(deadline_str, '%Y-%m-%d')
        except:
            return Response({'error': 'Deadline format invalid'}, status=status.HTTP_400_BAD_REQUEST)

        now = datetime.now()
        if deadline.date() < now.date():
            return Response({'error': 'Deadline is in the past'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            cvi = User.objects.get(id=cvi_id)
        except User.DoesNotExist:
            return Response({'error': 'CVI not found'}, status=status.HTTP_404_NOT_FOUND)
        if cvi.manager != user:
            
            return Response({'error': 'You are not authorized to make planning','id':user.id}, status=status.HTTP_403_FORBIDDEN)
        if remake:
            Visit.objects.filter(agent=cvi, validated=0).delete()


        if not cvi.agence:
            return Response({'error': 'CVI has no agence assigned'}, status=status.HTTP_400_BAD_REQUEST)

        number_of_days = (deadline.date() - now.date()).days
        daily_limit_minutes = 7 * 60
        total_time_minutes = number_of_days * daily_limit_minutes
        longitude = cvi.agence.longitude if not None else 31.610709890371677
        latitude = cvi.agence.latitude if not None else 2.8828559973535572
        start_point = {
            'name': 'Start',
            'longitude':longitude,
            'latitude':latitude
        }

        data_points = list(PointOfSale.objects.filter(manager=cvi).values())
        routes, edges, estimates = plan_multiple_days(
            [start_point] + data_points,
            daily_limit_minutes,
            speed_kmph
        )
        estimated_days=len(routes)
        warning = None
        if estimated_days>number_of_days:
            warning = "The CVI won't be able to visit all the points"
            suggestion='Increase the deadline'
            return Response({
            'message': 'Visits could not be scheduled',
            'visits_number': len(visits),
            'number_of_days': number_of_days,
            'estimated_days':estimated_days,
            'number_of_points':len(data_points),
            'suggestion':suggestion,
            'warning': warning
        }, status=status.HTTP_400_BAD_REQUEST)
        visits = []
        for day_index, route in enumerate(routes):
            for order_index, point_name in enumerate(route[1:], start=1):  # skip 'Start'
                pos = next((p for p in data_points if p['name'] == point_name), None)
                if pos:
                    visits.append(Visit(
                        id=uuid.uuid4(),
                        deadline=now + timedelta(days=day_index),
                        agent=cvi,
                        pdv_id=pos['id'],
                        status='scheduled',
                        order=order_index,
                        validated=0
                    ))

        Visit.objects.bulk_create(visits)
        cvi.deadline=deadline
        cvi.save()
        return Response({
            'message': 'Visits scheduled successfully',
            'visits_number': len(visits),
            'number_of_days': number_of_days,
            'number_of_points':len(data_points),
            'visits':routes,
            
            'warning': warning
        }, status=status.HTTP_201_CREATED)

        
        
class ValidatePlanning(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        data=request.data
        if not data.get('cvi'):
            return Response({'error': 'missing cvi'}, status=status.HTTP_400_BAD_REQUEST)
        user=request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role!='manager':
            return Response({'error': 'you can\'t validate visits'}, status=status.HTTP_400_BAD_REQUEST)
        visits=Visit.objects.filter(agent=data.get('cvi'),validated=0)
        for visit in visits:
            visit.validated=1
            visit.save()
        return Response({'message': 'Visits validated successfully'}, status=status.HTTP_201_CREATED)
    
    
    
    
class GetVisitsPlan(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user

        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'agent':
            return Response({'error': 'You can\'t get planning'}, status=status.HTTP_400_BAD_REQUEST)

        today = datetime.now().date()
        start_of_day = make_aware(datetime.combine(today, datetime.min.time()))
        end_of_day = make_aware(datetime.combine(today, datetime.max.time()))

        visits = Visit.objects.filter(
            agent=user,
            validated=1,
            status='scheduled',
            deadline__gte=start_of_day,
        ).order_by('deadline', 'order').select_related('pdv')

        # Group by deadline.date()
        grouped = defaultdict(list)
        for visit in visits:
            date_str = visit.deadline.date().isoformat()
            grouped[date_str].append({
                'visit_id': visit.id,
                'order': visit.order,
                'visit_time': visit.visit_time,
                'point_of_sale': {
                    'id': visit.pdv.id,
                    'name': visit.pdv.name,
                    'latitude': visit.pdv.latitude,
                    'longitude': visit.pdv.longitude
                }
            })

        # Prepare final sorted response
        sorted_data = []
        for date in sorted(grouped.keys()):
            sorted_data.append({
                'date': date,
                'visits': sorted(grouped[date], key=lambda x: x['order'])
            })

        return Response({'plan': sorted_data, 'id': user.id}, status=status.HTTP_200_OK)
        
        
    
    
    


class ClancelVisit(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        image=request.FILES.get('image')
        user = request.user
        if not user:
            return Response({'error': 'User not found'}, status=status.HTTP_400_BAD_REQUEST)
        if user.role != 'agent':
            return Response({'error': 'You can\'t cancel visit'}, status=status.HTTP_400_BAD_REQUEST)
        if not data.get('visit_id'):
            return Response({'error': 'Missing visit_id'}, status=status.HTTP_400_BAD_REQUEST)
        if not image:
            return Response({'error': 'Missing image'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            visit = Visit.objects.get(id=data.get('visit_id'))
        except Visit.DoesNotExist:
            return Response({'error': 'Visit not found'}, status=status.HTTP_400_BAD_REQUEST)
        if visit.agent != user:
            return Response({'error': 'You are not authorized to cancel visit'}, status=status.HTTP_400_BAD_REQUEST)
        if visit.status != 'scheduled':
            return Response({'error': 'Visit is not scheduled'}, status=status.HTTP_400_BAD_REQUEST)
        visit.status = 'cancelled'
        image_encoded= base64.b64encode(image.read())
        visit.cancel_proof=image_encoded
        visit.save()
        # reschedueling
        now = timezone.now()
        deadline=now
        visits=Visit.objects.filter(agent=user,status='scheduled',deadline__gte=now)
        data_points=[]
        pdv=visit.pdv
        data_points+=[{'id':pdv.id,'name':pdv.name,'longitude':pdv.longitude,'latitude':pdv.latitude}]
        for visit in visits:
            pdv=visit.pdv
            data_points.append({'id':pdv.id,'name':pdv.name,'longitude':pdv.longitude,'latitude':pdv.latitude})
            deadline=max(deadline,visit.deadline)
            visit.delete()
        start_point={'name':'Start','longitude':user.agence.longitude,'latitude':user.agence.latitude}
        number_of_days=(deadline-now).days
        daily_limit_minutes=7*60
        total_time_minutes=number_of_days*daily_limit_minutes
        routes,edges,estimates=plan_multiple_days([start_point]+data_points,total_time_minutes,daily_limit_minutes,60)
        visits = []
        for day_index, route in enumerate(routes):
            for order_index, point_name in enumerate(route[1:], start=1):  # skip 'Start'
                pos = next((p for p in data_points if p['name'] == point_name), None)
                if pos:
                    visits.append(Visit(
                        id=uuid.uuid4(),
                        deadline=now + timedelta(days=day_index+1),
                        agent=user,
                        pdv_id=pos['id'],
                        status='scheduled',
                        order=order_index,
                        validated=1
                    ))

        Visit.objects.bulk_create(visits)
        return Response({'message': 'Visit cancelled successfully'}, status=status.HTTP_201_CREATED)
        