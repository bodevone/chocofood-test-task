from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .algorithm import Algorithm


class IndexView(APIView):
    def post(self, request):
        review = request.data.get('text')
        algorithm = Algorithm()
        rating = algorithm.predict(review)
        data = {'rating': rating}
        return Response(data, status=status.HTTP_200_OK)