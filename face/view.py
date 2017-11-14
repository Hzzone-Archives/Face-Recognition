from django.http import HttpResponse
from django.shortcuts import render
from .models import Pairs
def main(request):
	context = {}
	context['pairs'] = [1,2,3]
	return render(request, 'main.html', context)