from django.urls import reverse
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
#from appCreditoBanco.Logica import modeloSNN
import pickle
import json
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
import numpy as np

class modeloSNN():
    """Clase modelo Preprocesamiento y SNN"""
    #Función para cargar preprocesador
    def cargarPipeline(self):
        pipe = Pipeline(steps=[])
        return pipe
    #Función para cargar red neuronal 
    def cargarNN(self,nombreArchivo):
        model = load_model(nombreArchivo+'.h5')    
        print("Red Neuronal Cargada desde Archivo") 
        return model

    def cargarTokenizer(self,nombreArchivo):
        with open(nombreArchivo+'.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return tokenizer

    def tokenizar(self, text):
        text= [text]
        tokenizer = self.cargarTokenizer(self,'Recursos/tokenizerPreprocesadores')
        x = tokenizer.texts_to_sequences(text)
        padded_sequences = pad_sequences(x, maxlen=81, padding='pre')
        x = pad_sequences(padded_sequences)
        x = pd.DataFrame(x)
        return x

    #Función para integrar el preprocesador y la red neuronal en un Pipeline
    def cargarModelo(self):
        #Se carga el Pipeline de Preprocesamiento
        pipe=self.cargarPipeline(self)
        print('Pipeline de Preprocesamiento Cargado')
        #Se carga la Red Neuronal
        modeloOptimizado=self.cargarNN(self,'Recursos/modeloRedNeuronalOptimizada')
        #Se integra la Red Neuronal al final del Pipeline
        pipe.steps.append(['modelNN',modeloOptimizado])
        cantidadPasos=len(pipe.steps)
        print("Cantidad de pasos: ",cantidadPasos)
        print(pipe.steps)
        print('Red Neuronal integrada al Pipeline')
        return pipe
    #Funcion para saber que tan bien estuvo la comida

    def predecirNUevoCliente(self,t = 'Si quieren comer comida Mexicana este no es el lugar. No merece encasillarla dentro de esa variedad, solo se lo gana por la .  muy mala, ni siquiera nos dieron la  canasta de nachos que le dieron a varias mesas. La comida vino antes que la bebida. Comida sin sabor, cara para lo que es y la carne hervida. No vuelvo jamas y no lo recomiendo ni a mi enemigo'):
        dic = {0:"Buena", 1:"Mala"}
        pipe = self.cargarModelo(self)
        Xnew = self.tokenizar(self,t)
        print(Xnew)
        pred = pipe.predict(Xnew)
        pred_labels = np.argmax(pred, axis=1)
        ClaseMayorProbabilidad=np.argmax(pred)
        prob = pred.tolist()[0][ClaseMayorProbabilidad]
        prob = str(round(prob*100, 4)) + '%'
        salida = {"clase":dic[int (pred_labels)],"certeza":prob }
        print(salida)
        return salida