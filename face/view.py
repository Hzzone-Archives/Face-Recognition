import predict
from django.shortcuts import render
from .models import Pairs
def main(request):
	context = {}
	context['pairs'] = predict.generate(totals=50, threshold=0.7301, features_source="face/features.txt")
	return render(request, 'main.html', context)