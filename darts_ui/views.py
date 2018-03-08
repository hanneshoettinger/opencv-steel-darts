from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse

from .models import StartDartsRecognition
# Create your views here.


def index(request):
    return HttpResponse("Darts Cam Status is: ()".format(
            "UNKNOWN"
        )
    )
