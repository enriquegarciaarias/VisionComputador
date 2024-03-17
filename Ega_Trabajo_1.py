#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visión por Computador
Ejercicio Trabajo 1
Author: Enrique Garcia

1) Leer la imagen a color.
2) Convertir a nivel de gris.
3) Mejorar la imagen en caso de ser necesario.
4) Umbralizar, resulta una imagen con fondo negro y con las letras en blanco.
5) Invertir para tener fondo blanco y letras en negro.
6) Buscar la palabra que desees usando OCR y opencv.
7) Probar con la imagen castellano_antiguo. Buscar alguna de sus palabras.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
import re
from pytesseract import Output


#
# ---------------------------------------------------------------------------------
# Descripción:	Modifica el tamañano de una imagen
#				manteniendo su aspecto. 
#				Utilizado con múltiples cv2.imshow
# Parámetros:	Imagen, ancho, alto, interpolación
# Retorno:		La imagen redimensionada
# ---------------------------------------------------------------------------------
def resize_imagen(i, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = i.shape[:2]
    if width is None and height is None:
        return i
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(i, dim, interpolation=inter)

#
# ---------------------------------------------------------------------------------
# Descripción:	Presenta varias imágenes
#				Da opción a escoger una imagen. 
# Parámetros:	Lista con las imágenes y textos descriptivos
# Retorno:		Imagen selecionada
# ---------------------------------------------------------------------------------
def escoger(li):
    ind = 1
    text = ""
    select = li[0]
    for ele in li:
        text += "\t" + "opcion " + str(ind) + ") " + ele[1] + "\n"
        li[ind - 1][1] = str(ind) + ") " + li[ind - 1][1]
        ind += 1
    show_img(li, 'plot')
    while True:
        print("Tiene " + str(len(li)) + " opciones")
        print(text)
        seleccion = input("----->Selección: ")
        if seleccion.isdigit():
            seleccion = int(seleccion)
            if seleccion < len(li) + 1:
                select = li[seleccion - 1]
                pattern = "[0-9]\) "
                select[1] = re.sub(pattern, "", select[1])
                break
            else:
                print("Opción no válida. Intente de nuevo.")
        else:
            print("Opción no válida. Indique un número")
    cv2.destroyAllWindows()
    return select

#
# ---------------------------------------------------------------------------------
# Descripción:	Muestra al operador las imágenes
# Parámetros:	Lista con las imágenes y textos descriptivos
#				Modo de presentación: CV o PLT
#				cmap: mapa de colores
#				wait: si debe esperar intervención operador
# Retorno:		Imagen selecionada
# ---------------------------------------------------------------------------------
def show_img(lista, mode="cv", cmap="gray", wait=None):
    if mode == 'cv':
        for item in lista:
            i = item[0]
            if wait is None:
                i = resize_imagen(item[0], width=640)
            cv2.imshow(item[1], i)
        if wait is None:
            return
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if mode == 'plot':
        count = len(lista)
        asignacion = [111, 121, 221, 221, 231, 231, 331, 331, 331]
        inicio = asignacion[count - 1]
        for item in lista:
            plt.subplot(inicio), plt.imshow(item[0], cmap), plt.title(item[1])
            plt.xticks([]), plt.yticks([])
            inicio += 1
        plt.show()

#
# ---------------------------------------------------------------------------------
# Descripción:	Conjunto de funciones erosion, dilatacion, apertura y cierre
#				erosion: operación morfológica de reducción cv2.erode
#				dilatacion: operación morfológica de ampliación cv2.dilate
#				apertura: erosion + dilatación
#				cierre: dilatación + erosión
# Parámetros:	i: imagen
#				sh: tamaño del kernel
#				it: iteraciones a aplicar
# Retorno:		Imagen transformada
# ---------------------------------------------------------------------------------
def erosion(i, sh, it):
    kernel = np.ones((sh, sh), np.uint8)
    return cv2.erode(i, kernel, iterations=it)


def dilatacion(i, sh, it):
    kernel = np.ones((sh, sh), np.uint8)
    return cv2.dilate(i, kernel, iterations=it)


def apertura(i):
    newimg = erosion(i, 4, 1)
    return dilatacion(newimg, 4, 1)


def cierre(i):
    newimg = dilatacion(i, 4, 1)
    return erosion(newimg, 4, 1)


def esqueleto(img):
    img_inv = cv2.bitwise_not(img)
    size = np.size(img_inv)
    skeleton = np.zeros(img_inv.shape, np.uint8)
    ret, img = cv2.threshold(img_inv, 127, 255, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    finished = False
    while not finished:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            finished = True
    return skeleton

#
# ---------------------------------------------------------------------------------
# Descripción:	Se aplican varias operaciones
# Parámetros:	imagen a procesar
# Retorno:		Imagen procesada
# ---------------------------------------------------------------------------------
def eliminar_Ruido(i):
    p = 1 / 9
    kernel = np.asarray([[p, p, p], [p, p, p], [p, p, p]])
    paso_bajo = cv2.filter2D(i, -1, kernel)
    paso_bajo = cv2.convertScaleAbs(paso_bajo)
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(i, -1, kernel)
    blur = cv2.blur(i, (3, 3))
    median = cv2.medianBlur(i, 5)
    gblur = cv2.GaussianBlur(i, (5, 5), 0)
    return paso_bajo, dst, blur, gblur, median


def realce_bordes(img):
    kernel = np.ones((5, 5), np.float32) / 30
    # filtro paso alto o box filter
    filter2d = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    # filtro paso bajo Fourrier
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    espectro = np.log(np.abs(fshift))
    radio = 20
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - radio:crow + radio, ccol - radio:ccol + radio] = 1
    f_ishift = np.fft.ifftshift(fshift)
    img_filtro = np.fft.ifft2(f_ishift)
    img_filtro = np.abs(img_filtro)
    # Detector de Canny
    edges = cv2.Canny(img, 100, 200)
    # Operador Sobel
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    map, ang = cv2.cartToPolar(gx, gy)
    abs_grad_x = cv2.convertScaleAbs(gx)
    abs_grad_y = cv2.convertScaleAbs(gy)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    lista = [[filter2d, 'Box Filter'],
             [espectro, 'Paso Bajo Fourrier'],
             [img_filtro, 'Filtro'],
             [edges, 'Filter Canny'],
             [grad, 'Sobel']]
    return lista


def Aumenta_contraste(i):
    result = cv2.equalizeHist(i)
    return result


def normalizar(i):
    norm_img = np.zeros((i.shape[0], i.shape[1]))
    norm_img = cv2.normalize(i, norm_img, 0, 255, cv2.NORM_MINMAX)
    return norm_img

#
# ---------------------------------------------------------------------------------
# Descripción:	Lectura y conversión a escala de grises
# Parámetros:	imagen a procesar
# Retorno:		Imagen procesada
# ---------------------------------------------------------------------------------
def leer_convertir(i):
    img = cv2.imread(i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

#
# ---------------------------------------------------------------------------------
# Descripción:	Secuencia de procesos de mejora de imagen
# Parámetros:	Imagen a procesar
# Retorno:		Imagen procesada y seleccionada
# ---------------------------------------------------------------------------------
def mejorar_imagen(i):
    img = normalizar(i)
    lista = []
    lista.append([img, 'Original'])
    lista += realce_bordes(img)
    select = escoger(lista)
    lista = []
    lista.append(select)
    r2, r3, r4, r5, r6 = eliminar_Ruido(img)
    lista.append([r2, 'Paso Bajo'])
    lista.append([r3, 'Promediada'])
    lista.append([r4, 'Blur'])
    select = escoger(lista)
    lista = []
    lista.append(select)
    lista.append([r5, 'GBlur'])
    lista.append([r6, 'Median'])
    r7 = apertura(img)
    lista.append([r7, 'Apertura'])
    select = escoger(lista)
    lista = []
    lista.append(select)
    r8 = cierre(img)
    lista.append([r8, 'Cierre'])
    r9 = esqueleto(img)
    lista.append([r9, 'Esqueleto'])
    r10 = Aumenta_contraste(img)
    lista.append([r10, 'Contraste'])
    select = escoger(lista)
    print("La imagen escogida es: " + select[1])
    return select[0]


def umbralizacion_binaria(img):
    lista = []
    edges = cv2.Canny(img, 10, 100, apertureSize=3)
    lista.append([edges, 'Umbral. Bordes'])
    return lista


def umbralizacion_frontera(img):
    lista = []
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    filas, columnas = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    abs_grad_x = cv2.convertScaleAbs(gx)
    abs_grad_y = cv2.convertScaleAbs(gy)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    u = 30
    color = 3
    i_zeros = np.zeros([filas, columnas, color])
    for f in range(filas):
        for c in range(columnas):
            if grad[f][c] < u:
                i_zeros[f][c][0] = 150
            elif laplacian[f][c] >= 0:
                i_zeros[f][c][1] = 150
            else:
                i_zeros[f][c][2] = 150

    lista.append([grad, 'Umbral. Sobel'])
    lista.append([laplacian, 'Umbral. Laplaciana'])
    lista.append([i_zeros.astype("uint8"), 'Umbral. Frontera'])
    return lista


def umbralizar_invertir(img):
    lista = []
    lista.append([img, 'Original'])
    lista += umbralizacion_binaria(img)
    lista += umbralizacion_frontera(img)
    select = escoger(lista)
    if select[1] == "Original":
        return img
    return cv2.bitwise_not(select[0])

#
# ---------------------------------------------------------------------------------
# Descripción:	Localiza y presenta un texto en una imagen
#				Indica el número de apariciones
# Parámetros:	i: Imagen, d:directorio operación
# ---------------------------------------------------------------------------------
def buscar_texto(i, d):
    cv2.imwrite(d + "temp.png", i)
    data_image = pytesseract.image_to_data(i, output_type=Output.DICT)
    color = (0, 0, 255)
    n_boxes = len(data_image['text'])
    while True:
        print("Indique texto a buscar. Teclee 0 para salir")
        texto = input("------>Elección: ")
        cv2.destroyAllWindows()
        if texto == "0":
            break
        num_palabras = 0
        imgtext = cv2.imread(d + "temp.png", 0)
        for j in range(n_boxes):
            if int(data_image['conf'][j]) > 40:
                match = re.match(texto, data_image['text'][j])
                if match:
                    num_palabras += 1
                    (x, y, w, h) = (
                        data_image['left'][j], data_image['top'][j], data_image['width'][j], data_image['height'][j])
                    imgtext = cv2.rectangle(imgtext, (x, y), (x + w, y + h), color, 2)

        show_img([[imgtext, 'OCR']], 'plot')
        print("Se han encontrado " + str(num_palabras) + " coincidencias")


# *************************************************************
# Programa Base
#
print()
print("\tUNED",
      "\tTrabajo 1 Visión por Computador",
      "\t________________________________________________",
      sep="\n")
directorio = 'Imagenes_Trabajo_1/'
imagenes = ["castellano_antiguo.png", "texto_delfines.png"]
for image in imagenes:
    img = leer_convertir(directorio + image)
    img = mejorar_imagen(img)
    print("\t________________________________________________",
          "\tProceso de Umbralizar e Invertir ",
          sep="\n")
    img = umbralizar_invertir(img)
    buscar_texto(img, directorio)
print("\t________________________________________________",
      "\tSu proceso ha finalizado ",
      sep="\n")
# *************************************************************
