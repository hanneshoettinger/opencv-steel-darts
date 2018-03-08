from datetime import datetime
from django.db import models
# from .darts_recognition.Start import kickoff
# Create your models here.


class StartDartsRecognition(models.Model):
    last_updated_date = models.DateTimeField('date updated')
    is_automated_darts_recognition_started = models.BooleanField()

    def __str__(self):
        now = datetime.now()
        currentstatus = "{}: Running: {} set at {}".format(
            now,
            self.is_automated_darts_recognition_started,
            self.last_updated_date
        )
        return currentstatus
