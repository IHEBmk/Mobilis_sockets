"""
URL configuration for mobilis_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from server.CoordinatesView import GetCoordinates, RefreshCoordinates
from server.PerformanceView import getGlobalPerformancePDV, getVisitPerformance
from server.UserViews import AssignZoneView, LoginView, SignupView
from server.CommuneView import Commune_to_supabase
from server.PdvViews import AddPdv, DeletePdv, GetPdv, Pdv_To_supabase, UpdateStatusPdv, VisitPdv
from server.WilayaView import Wilaya_to_supabase
from rest_framework_simplejwt.views import TokenRefreshView




urlpatterns = [
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('assign/', AssignZoneView.as_view(), name='assignzonetouser'),
    path('pdv_to_supa/', Pdv_To_supabase.as_view(), name='pdv_to_supa'),
    path('wilaya_to_supa/', Wilaya_to_supabase.as_view(), name='wilaya_to_supa'),
    path('commune_to_supa/', Commune_to_supabase.as_view(), name='commune_to_supa'),
    path('GetPdv/', GetPdv.as_view(), name='GetPdv'),
    path('UpdateStatusPdv/', UpdateStatusPdv.as_view(), name='UpdateStatusPdv'),
    path('VisitPdv/', VisitPdv.as_view(), name='VisitPdv'),
    path('DeletePdv/', DeletePdv.as_view(), name='DeletePdv'),
    path('GetCoordinates/', GetCoordinates.as_view(), name='GetCoordinates'),
    path('RefreshCoordinates/', RefreshCoordinates.as_view(), name='RefreshCoordinates'),
    path('getGlobalPerformancePDV/', getGlobalPerformancePDV.as_view(), name='getGlobalPerformancePDV'),
    path('getVisitPerformance/', getVisitPerformance.as_view(), name='getVisitPerformance'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]







