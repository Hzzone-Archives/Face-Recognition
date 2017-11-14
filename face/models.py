from django.db import models
class Pairs(models.Model):
    first_image = models.CharField(max_length=100)
    second_image = models.CharField(max_length=100)
    real_result = models.CharField(max_length=100)
    predict_result = models.CharField(max_length=100)