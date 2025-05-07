# UserViews.py
from datetime import datetime
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils import timezone
from server.models import Commune, PointOfSale, User, Visit, Zone
from .Userserializers import UserAssignSerializer, UserLoginSerializer, UserSignupSerializer
from rest_framework.permissions import IsAuthenticated
from .authentication import CustomJWTAuthentication, get_tokens_for_user
from django.db.models import Count

class getGlobalPerformancePDV(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # Get the user's performance data
        user = request.user
        data = request.query_params
        page_size = int(data.get('page_size', 10))
        page_num = int(data.get('page', 1))
        data=request.query_params
        if not user:
            return Response({'message': 'Unauthorized'}, status=status.HTTP_401_UNAUTHORIZED)
        if not data.get('from') or not data.get('to'):
            return Response({"message":"missing input"}, status=status.HTTP_400_BAD_REQUEST)
        from_= datetime.strptime(data.get('from'), '%Y-%m-%d')
        to_= datetime.strptime(data.get('to'), '%Y-%m-%d')
        from_ = timezone.make_aware(from_, timezone.get_current_timezone())
        to_ = timezone.make_aware(to_, timezone.get_current_timezone())
        if user.role == 'admin':
            # Get all the pdv created <= to_ and find them in visit
            total_visits=list(Visit.objects.filter(deadline__gte=from_, deadline__lte=to_).values())
            users=list(User.objects.filter(role='agent').values('id','first_name','last_name'))
        elif user.role == 'manager':
            communes=list(Commune.objects.filter(wilaya=user.wilaya).values('id'))
            pdvs=list(PointOfSale.objects.filter(commune__in=communes, created_at__lte=to_).values())
            total_visits=list(Visit.objects.filter(deadline__gte=from_, deadline__lte=to_,pdv__in=pdvs).values())
            users=list(User.objects.filter(role='agent', manager=user.id).values('id','first_name','last_name'))
        elif user.role == 'agent':
            pdvs=list(PointOfSale.objects.filter(created_at__lte=to_, manager=user.id).values())
            total_visits=list(Visit.objects.filter(deadline__gte=from_, deadline__lte=to_,pdv__in=pdvs).values())
        if len(total_visits) == 0:
            percentage_visit=0
            #return Response({'message':'No visits found for timestamp'}, status=status.HTTP_404_NOT_FOUND)
        else:
            total =len(total_visits)
            
            count=0
            for visit in total_visits:
                if visit['status']=='finished':
                    count+=1
            
            percentage_visit=round(count/total*100,2)
        if users:
            
            top_visitors = list((
                        Visit.objects
                        .filter(visit_time__gte=from_, visit_time__lte=to_,status='finished')
                        .values("agent")
                        .annotate(visit_count=Count("id"))
                        .order_by("-visit_count")  # Sort descending
                    ))
            if len(users)<page_num*page_size:
                Response({'message':'success',
                    'percentage_visit': percentage_visit,
                    'users_performance': users if users else None,}
                , status=status.HTTP_200_OK)
            else:
                for user in users:
                    
                    user['objective']=PointOfSale.objects.filter(manager=user['id']).count()
                    user['visited_number']= next((item['visit_count'] for item in top_visitors if item['agent'] == user['id']), 0)
                    user['performance']=round(user['visited_number']/user['objective']*100,2) if user['objective']>0 else 0
            
            return Response(
            {'message':'success',
                'percentage_visit': percentage_visit,
                'users_performance': users[page_num*page_size-page_size:page_num*page_size] if users else None,}
            , status=status.HTTP_200_OK
        )
        
            
            
            



class getVisitPerformance(APIView):
    authentication_classes = [CustomJWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        # Get the user's performance data
        user = request.user
        data=request.query_params
        if not user:
            return Response({'message': 'Unauthorized'}, status=status.HTTP_401_UNAUTHORIZED)
        if not data.get('from') or not data.get('to'):
            return Response({"message":"missing input"}, status=status.HTTP_400_BAD_REQUEST)
        from_= datetime.strptime(data.get('from'), '%Y-%m-%d')
        to_= datetime.strptime(data.get('to'), '%Y-%m-%d')
        from_ = timezone.make_aware(from_, timezone.get_current_timezone())
        to_ = timezone.make_aware(to_, timezone.get_current_timezone())
        if user.role == 'admin':
            visits=list(Visit.objects.filter(visit_time__gte=from_, visit_time__lte=to_).values())
        elif user.role == 'manager':
            communes=list(Commune.objects.filter(wilaya=user.wilaya).values('id'))
            pdvs=list(PointOfSale.objects.filter(commune__in=communes, created_at__lte=to_).values())
            visits=list(Visit.objects.filter(visit_time__gte=from_, visit_time__lte=to_, pdv__in=pdvs).values())
        elif user.role == 'agent':
            visits=list(Visit.objects.filter(visit_time__gte=from_, visit_time__lte=to_, agent=user.id).values())
        total_visits=len(visits)
        mean_time=0
        finish_rate=0
        if total_visits!=0:    
            for visit in visits:
                
                if visit['status']=='finished':
                    mean_time+=visit['duration']
                    finish_rate+=1
            mean_time=mean_time/total_visits
        for visit in visits:
            agent = visit['agent_id']  # This should be the user ID of the agent
            user_data = User.objects.get(id=agent)  # Fetch the User instance
            
            # Add all user attributes to the visit dictionary
            visit.update({
                'user_name':user_data.last_name if user_data.last_name else ""+' '+user_data.first_name if user_data.first_name else "",
            })
        return Response(
            {'message':'success',
                'mean_time': mean_time,
                'finish_rate': finish_rate,
                'total_visits': total_visits,
                'details': visits if visits else None,},
            status=status.HTTP_200_OK
        )
        
        
        
        