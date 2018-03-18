from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect, request
from django.urls import reverse
from django.views import generic

from darts_ui.models import DartsRecognitionStatus
from darts_ui.darts_recognition.Start import return_status
from datetime import datetime
# Create your views here.


# class IndexView(generic.ListView):
#     template_name = 'darts_ui/index.html'
#     context_object_name = "status"
#
#     def start_stop(self, request):
#         if request.GET.get('start'):
#             try:
#                 started = return_status()
#                 if started:
#                     s = DartsRecognitionStatus.objects.last()
#                     s.is_running = True
#                     s.since = datetime.now()
#             except:
#                 pass
#         else:
#             try:
#                 stopped = return_status()
#                 if stopped:
#                     s = DartsRecognitionStatus.objects.last()
#                     s.is_running = True
#                     s.since = datetime.now()
#             except:
#                 pass
#
#
#     def get_queryset(self):
#         """
#         Display the current status of the darts recognition
#         """
#         current = DartsRecognitionStatus.objects.last()
#         # return {"status": current, "since": since}
#         return current
#
#
#



def index(request):
    status = DartsRecognitionStatus.objects.last()
    if request.method == 'POST':
        try:
            print(return_status())
            if return_status():
                new_status = DartsRecognitionStatus(is_running=(not status.current_status()),since=datetime.now())
                new_status.save()
                print("Trying to render new page")
                return render(request, 'darts_ui/index.html', {"status": new_status.is_running, "since": new_status.since})
        except:
            pass

    if status.current_status():
        return render(request, 'darts_ui/index.html', {"status": True, "since": status.since})
    else:
        return render(request, 'darts_ui/index.html', {"status": False, "since": status.since})
