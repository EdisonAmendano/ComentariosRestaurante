from django.shortcuts import render
from appComentariosRestaurante.Logica import modeloSNN
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import json
from django.http import JsonResponse

class Clasificacion():
    def determinarAprobacion(request):
        return render(request, "VistaComentarios.html")
    
    @api_view(['GET','POST'])
    def predecir(request):
        try:
            #Formato de datos de entrada
            TEXTO = str(request.POST.get('Texto'))
            #Consumo de la lógica para predecir si se aprueba o no el crédito
            resul=modeloSNN.modeloSNN.predecirNUevoCliente(modeloSNN.modeloSNN,TEXTO)
        except:
            resul='Datos inválidos'
        return render(request, "Informe.html",{"e":resul})
    
