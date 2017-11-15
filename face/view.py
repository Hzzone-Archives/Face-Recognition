import predict
from django.shortcuts import render
from .models import Pairs
def main(request):
	context = {}
	context['pairs'] = predict.generate()
	return render(request, 'main.html', context)