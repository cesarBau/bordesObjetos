"""
@author: Cesar
"""
import cv2

# Cargamos la imagen
original = cv2.imread("img6.jpeg")
# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (3, 3), 0)
# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)
# Operaciones Morfologicas Cierre
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imshow("Closed", closed)
# Buscamos los contornos
(contornos, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
for c in contornos:
    area = cv2.contourArea(c)
    if area > 2000:
        total = total + 1
        # Calculamos los momentos invariantes de hu
        m = cv2.moments(c)
        mHU = cv2.HuMoments(m).flatten()
        # Encontramos el centro de la figura
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        centro = cx, cy
        # Mostramos los 7 momentos invariantes de hu
        print("Momento de hu figura " + str(total) + " valores: ")
        print(str(mHU))
        # Dibujamos el contorno
        cv2.drawContours(original, [c], -1, (0, 255, 0), 2, cv2.LINE_AA)
        letrero = 'Obj: ' + str(total)
        cv2.putText(original, letrero, centro, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
# Mostramos el número de monedas por consola
print("He encontrado {} objetos".format(len(contornos)))
cv2.imshow("contornos", original)
cv2.waitKey(0)
