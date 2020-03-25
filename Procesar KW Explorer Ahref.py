#!/usr/bin/env python
# coding: utf-8
# Author: Jlmarin
# Web: https://jlmarin.eu

import argparse
import sys
import pandas as pd
from nltk import SnowballStemmer
import spacy
import es_core_news_sm
from tqdm import tqdm
from unidecode import unidecode
import glob
import re

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save', help='Nombre del archivo al guardar')
parser.add_argument('-c', '--clean', nargs='?', const=1, type=bool, default=True, help='Elimina todos los duplicados del listado')
parser.add_argument('-i', '--intent', nargs='?', const=1, type=bool, default=True, help='Activa el procesado de las intenciones de busqueda')
parser.add_argument('-l', '--location', nargs='?', const=1, type=bool, default=True, help='Nombre del archivo con la base de datos de las localizaciones')
args = parser.parse_args()

pd.options.mode.chained_assignment = None 

nlp = es_core_news_sm.load()
spanishstemmer=SnowballStemmer('spanish')

def normalize(text):
    text = unidecode(str(text))
    doc = nlp(text)
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if len(t) > 3 and t.isalpha()]
    
    return lexical_tokens

def raiz(kw):
    #Calculamos la raiz semantica
    stems = [spanishstemmer.stem(wd) for wd in kw]
    raiz = " ".join(sorted(stems))
    return raiz

# Abrimos todos los archivos CSV y los agregamos a un dataframe    
archivos=[]
files = glob.glob("entrada/*.csv")

loop = tqdm(total = len(files), position = 0, leave = False)
for f in files:
    loop.set_description("Unificando archivos...".format(f))  
    archivos.append(pd.read_csv(f))
    loop.update(1)

df=pd.concat(archivos,ignore_index='True')  
loop.close()
print('Archivos cargados... OK')

# Eliminamos duplicados
if args.clean:
    df = df.drop_duplicates() 
    print('Duplicados eliminados... OK')

# Bucle principal de procesado
loop = tqdm(total = len(df.index), position = 0, leave = False)

df['Raiz semantica'] = ''

print(df)
for i in df.index:
    loop.set_description("Calculando raices...".format(i))
    
    kw_a = normalize(df.loc[i,'Keyword'])
    
    #Calculamos la raiz semantica
    df.loc[i,'Raiz semantica'] = raiz(kw_a)
    
    loop.update(1)
    #print('Kw ' + str(index) + ' de ' +  str(len(df.index)))
loop.close()
print('Calculado raices semanticas... OK')
df = df.sort_values(by=['Raiz semantica', 'Volume'], ascending=[True,False])
df = df.reset_index(drop=True)

# Agrupamos las keywords segun su raiz semantica y el volumen de busquedas
loop = tqdm(total = len(df.index), position = 0, leave = False)

df['Grupo'] = ''
for i in df.index:
    loop.set_description("Agrupando...".format(i))
    if i == 0:
        df.loc[i,'Grupo'] = df.loc[i,'Keyword']
    elif df.loc[i,'Raiz semantica'] == df.loc[i-1,'Raiz semantica']:
        df.loc[i,'Grupo'] = df.loc[i-1,'Grupo']
    else:
        df.loc[i,'Grupo'] = df.loc[i,'Keyword']
        
    loop.update(1)
    
loop.close()
print('Agrupado... OK')

df.to_csv('kw_procesado.csv', index=False)
print('Archivo kw_procesado.csv creado... OK')
gdf = (df.groupby('Grupo', as_index=False)
    .agg({'Volume':'sum','Clicks':'sum','Difficulty':'mean','CPC':'mean','CPS':'mean','Return Rate':'mean','Keyword':' | '.join}))

# Detectamos la intencion de busqueda de la kw: Informacional, transacional, navegacional
if args.intent:
    intenciones = pd.read_csv('Data/intenciones.csv')

    loop = tqdm(total = len(intenciones.index), position = 0, leave = False)

    gdf['Intencion'] = ''
    for i in intenciones.index:
        loop.set_description("Detectando intenciones de busqueda...".format(i))

        row = gdf[gdf['Grupo'].str.match(str(intenciones.loc[i,'Patron']))]

        if row is not None:
            gdf.loc[row.index,'Intencion'] = intenciones.loc[i,'Tipo']
        

        loop.update(1)

    loop.close()
    print('Intenciones de busqueda... OK')

# Detectamos la ubicacion de la palabra clave.
if args.location:
    ubicaciones = pd.read_csv('Data/ubicaciones.csv')

    loop = tqdm(total = len(ubicaciones.index), position = 0, leave = False)

    gdf['Ubicacion'] = ''
    gdf['Tipo ubicacion'] = ''
    for i in ubicaciones.index:
        loop.set_description("Detectando ubicaciones...".format(i))

        row = gdf[gdf['Grupo'].str.match(str(ubicaciones.loc[i,'Ubicacion']))]

        if row is not None:
            gdf.loc[row.index,'Ubicacion'] = ubicaciones.loc[i,'Ubicacion']
            gdf.loc[row.index,'Tipo ubicacion'] = ubicaciones.loc[i,'Tipo']
        
        loop.update(1)

    loop.close()
    print('Ubicaciones... OK')

gdf.to_csv('kw_agrupado.csv',index=False)
print('Archivo kw_agrupado.csv creado... OK')
print('Proceso finalizado... OK')
