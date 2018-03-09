from datetime import datetime
from django.db import models
from .darts_recognition.Start import kickoff
# Create your models here.


class DartsRecognitionStatus(models.Model):
    is_running = models.BooleanField()
    since = models.DateTimeField('date started')

    def __str__(self):
        current = self.objects.get(pk=1)
        if current.is_running:
            return "Status: {} since {}".format(current.is_running, current.since)
        else:
            return "{}".format(current.is_running)

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
        return self.objects.get(pk=1).is_running
