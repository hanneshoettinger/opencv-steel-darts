from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect, request
from django.urls import reverse
from django.views import generic

from darts_ui.models import DartsRecognitionStatus
from darts_ui.darts_recognition.Start import return_status
from datetime import datetime
# Create your views here.


def index(request):
    # print("Request: {}".format(request))
    status = DartsRecognitionStatus.objects.last()
    if request.method == 'POST':
        try:
            if return_status():
                new_status = DartsRecognitionStatus(is_running=(not status.current_status()), since=datetime.now())
                new_status.save()
                return render(request,
                              'darts_ui/index.html',
                              {"status": new_status.is_running, "since": new_status.since}
                              )
        except:
            pass

    if status.current_status():
        return render(request, 'darts_ui/index.html', {"status": True, "since": status.since})
    else:
        return render(request, 'darts_ui/index.html', {"status": False, "since": status.since})
