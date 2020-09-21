from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .algorithm import Algorithm


class IndexView(APIView):
    def __init__(self):
        self.algorithm = Algorithm()

    def post(self, request):
        review = request.data.get('text')
        rating = self.algorithm.predict(review)
        data = {'rating': rating}
        return Response(data, status=status.HTTP_200_OK)