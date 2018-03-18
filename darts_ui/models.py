from datetime import datetime
from django.db import models
from darts_ui.darts_recognition.Start import kickoff, return_status
# Create your models here.


class DartsRecognitionStatus(models.Model):
    is_running = models.BooleanField()
    since = models.DateTimeField('date started')

    def __str__(self):
        return "{}, {}".format(self.is_running, self.since)
        #
        # if current.is_running:
        #     return "Status: {} since {}".format(current.is_running, current.since)
        # else:
        #     return "{}".format(current.is_running)

    # TODO: Create methods to start and stop the dart recognition
    # def start_recognition(self):
    #     current = self.objects.get(pk=1)
    #     if not current.is_running:
    #         try:
    #             kickoff()
    #             current.objects.get(pk=1).update(is_running=True)
    #             current.objects.get(pk=1).update(running_since=datetime.now())
    #             return "Recognition started at {}.".format(current.since)
    #         except:
    #             raise EnvironmentError("Failed to start darts recognition")
    #     else:
    #         return "Recognition running since {}.".format(current.since)
    #
    # def stop_recognition(self):
    #     if self.is_running:
    #         try:
    #             # TODO: kickoff() needs a way to stop
    #             self.objects.get(pk=1).update(is_running=False)
    #             self.objects.get(pk=1).update(running_since=datetime.now())
    #             return "Recognition stopped at {}".format(self.since)
    #         except:
    #             raise ProcessLookupError("No way to turn off darts recognition")
    #     else:
    #         return "Recognition turned off since {}.".format(self.since)

    def current_status(self):
        return self.objects.last().is_running

    def running_since(self):
        return self.objects.last().since

    def start_recognition(self):
        return return_status()
