# asl_site\urls.py

from django.contrib import admin
from django.urls import path, include
from translator.views import index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('translator.urls')),
]