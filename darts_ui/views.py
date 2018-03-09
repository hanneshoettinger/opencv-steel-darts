from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse

from .models import DartsRecognitionStatus
# Create your views here.


def index(request):
    status = DartsRecognitionStatus.objects.last()
    if not status:
        return HttpResponse("Not running since {}.".format(status.since))
    else:
        return HttpResponse("Running since {}.".format(status.since))
