# Alvaro Garcia Gutierrez
# Elena Maria Perez Perez
# Henar Roman Serna

import math
from pyrobot.brain import Brain

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from imageio import imread
import joblib

# Text
font = cv2.FONT_HERSHEY_SIMPLEX
org = (15, 15)
org_2 = (15,30)
org_3 = (15,45)
org_4 = (15,60)
org_5 = (15,75)
fontScale = 0.5
color = (0, 0, 0)
thickness = 1

class BrainFollowLine(Brain):

	# Variables
	status = 0
	arrow = 2
	acc = [0,0,0,0]
	acc_mark = [0,0,0,0]
	im_border = np.zeros((240, 320), dtype = np.bool)
	clf = None

	def findLineDeviation(self, im):
		# Recorto la imagen
		#im = im[90:,:]
    		
		# Estudiamos en que caso nos encontramos
		img_cont, case, cent = define_case(im, self.im_border)
		
		if case == -1:
			return False, None, None, img_cont
		
		# Estudiamos las marcas del fotograma
		d = "None" 
		if case < 3:
			self.arrow = 2
			self.acc = [0,0,0,0]
			img_cont, self.acc_mark = study_marks(img_cont, im, self.acc_mark, self.clf)
		else:
			img_cont, d, self.arrow, self.acc = study_arrow(img_cont, im, case, self.arrow, [0,0,0,0])    
			#img_cont, d, self.arrow, self.acc = study_arrow(img_cont, im, case, self.arrow, self.acc)    
			text = "Flecha: " + d 
			cv2.putText(img_cont, text, org_2, font, fontScale, color, thickness, cv2.LINE_AA)
			self.acc_mark = [0,0,0,0]
		
		# Estudiamos la salida elegida
		img_cont, forwardVelocity, turnSpeed = eval_exit(img_cont, cent, self.arrow)
		
		return True, forwardVelocity, turnSpeed, img_cont

	def step(self):
		# take the last image received from the camera and convert it into opencv format
		try:
			cv_image = self.bridge.imgmsg_to_cv2(self.rosImage, "bgr8")
		except CvBridgeError as e:
			print(e)

		# determine the robot's deviation from the line.
		foundLine, forwardVelocity, turnSpeed, final_image = self.findLineDeviation(cv_image)
		
		# display the robot's camera's image using opencv
		cv2.imshow("Stage Camera Image", cv_image)
		cv2.waitKey(1)
		
		# get the distance from the sensors
		front = min([s.distance() for s in self.robot.range["front"]])
		left = min([s.distance() for s in self.robot.range["left-front"]])
		right = min([s.distance() for s in self.robot.range["right-front"]])
		
		#Tests
		#self.move(forwardVelocity, turnSpeed)
		#return

		# Searching a line or a wall
		if self.status == 0:
			if front < 0.5:
				#print("pared encontrada de frente, pasamos al estado 2")
				self.status = 2
				self.move(0.05, -0.5)
			elif foundLine:
				#print("linea encontrada, pasamos al estado 1")
				self.status = 1
				self.move(forwardVelocity, turnSpeed)
			else:
				#print("Sigo en el estado 0 buscando una pared o una línea")
				self.move(0.5, 0)

		# Following a line
		elif self.status == 1:
			if front < 0.5 or left < 0.4 or right < 0.4:
				#print("demasiado cerca de la pared, pasamos al estado 2")
				self.status = 2
				self.move(0.05, -0.5)
			elif foundLine:
				#print("siguiendo linea")
				self.move(forwardVelocity, turnSpeed)
			else:
				#print("linea perdida, pasamos al estado 3")
				self.status = 3
				self.move(0.2, 0.0)

		# Following a wall
		elif self.status == 2:
			if front < 0.8:
				#print("demasiado cerca de la pared de delante, giro a la derecha")
				self.move(0.05, -0.5)
			elif left < 0.6:
				#print("demasiado cerca de la pared izquierda, giro a la derecha")
				self.move(0.05, -0.5)
			elif right < 0.6:
				#print("demasiado cerca de la pared derecha, giro a la izquierda")
				self.move(0.05, 0.5)
			elif foundLine: # and front > 0.7 and left > 0.7 and right > 0.7
				#print("linea encontrada, pasamos al estado 1")
				self.status = 1
				self.move(0.05, -1)
			elif left > 0.6:
				#print("izquierda libre, giro a la izquierda")
				self.move(0.2, 0.5)
			elif front > 0.5:
				#print("frente libre, sigo recto")
				self.move(0.5, 0)

		# Searching the line in circles
		elif self.status == 3:
			if front < 0.5 or left < 0.5:
				#print("pared encontrada, pasamos al estado 2")
				self.status = 2
				self.move(0, -0.2)
			elif foundLine:
				#print("linea encontrada, pasamos al estado 1")
				self.status = 1
				self.move(forwardVelocity, turnSpeed)
			else:
				#print("Sigo en el estado 3 buscando una pared o una línea")
				self.move(0.2, 0.5)
				
		
	def setup(self):
		# Basics
		self.image_sub = rospy.Subscriber("/image",Image,self.callback)
		self.bridge = CvBridge()
		
		# Im_border
		for i in range(240):
			for j in range(320):
				if i<3 or i>237 or j<3 or j>317:
					self.im_border[i,j] = True
				else:
					self.im_border[i,j] = False
					
		# cargar el clf
		self.clf = joblib.load('./clf/clf.pkl')
		

	def callback(self,data):
		self.rosImage = data

	def destroy(self):
		cv2.destroyAllWindows()


def INIT(engine):

	assert (engine.robot.requires("range-sensor") and engine.robot.requires("continuous-movement"))

	return BrainFollowLine('BrainFollowLine', engine)
	
	
	
def define_case(im, im_border):

	# Separo los bordes y extraigo los contornos
	img_grey_line = (np.sum(im==np.array([255,0,0]),axis=2)==3).astype("uint8")*255
	border = np.logical_and(img_grey_line, im_border).astype("uint8")*255
	contList,hier = cv2.findContours(border,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
	# Pinto los contornos
	#img_cont = cv2.drawContours(np.float32(im), contList, -1, (0,255,0))
	img_cont = cv2.drawContours(np.float32(im), [], -1, (0,255,0))
	    
	# Calculolos centros de los contornos
	cent = []
	for c in contList:
		M = cv2.moments(c)
		try:
			center = [M["m10"]/M["m00"], M["m01"] / M["m00"]]
			cent.append(center)
		except:
			#print("Excepción")
			a=1
			
		
	# Pintamos las salidas y la entrada
	for c in cent:
		img_cont = cv2.circle(img_cont, (int(c[0]),int(c[1])), 8, (0,32,255), 15)
    
	# Defino el caso en funcion del numero de contornos
	n = len(cent)
	case = -1

	if n > 1:
		if n < 3:
			d = [cent[0][0]-cent[1][0], cent[0][1]-cent[1][1]]
			if d[0] < -50:
				cv2.putText(img_cont, 'Curva a la derecha', org, font, fontScale, color, thickness, cv2.LINE_AA)
				case = 1 
			elif d[0] > 50:
				cv2.putText(img_cont, 'Curva a la izquerda', org, font, fontScale, color, thickness, cv2.LINE_AA)
				case = 0 
			else:
				cv2.putText(img_cont, 'Linea recta', org, font, fontScale, color, thickness, cv2.LINE_AA)
				case = 2
		elif n < 4:
			cv2.putText(img_cont, 'Bifurcacion en Y', org, font, fontScale, color, thickness, cv2.LINE_AA)
			case = 3
		elif n < 5:
			cv2.putText(img_cont, 'Cruce en X', org, font, fontScale, color, thickness, cv2.LINE_AA)
			case = 4
		
	return img_cont, case, cent


def study_arrow(im, img_seg, case, arrow, acc):
    
	d = "None"
    
	# Construyo imágenes a partir de las etiquetas de segmentación y extraigo los contornos
	img_grey_mark = (np.sum(img_seg==np.array([0,0,255]),axis=2)==3).astype("uint8")*255 
	contList,hier = cv2.findContours(img_grey_mark,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
	# Cojo el contorno mas grande con un área mayor a 200
	if len(contList) != 0:
		contList = [max(contList, key = cv2.contourArea)]
		area = cv2.contourArea(contList[0])
		
		if area < 200:
			contList = []
	    
	# Pinto los contornos
	img_cont = cv2.drawContours(np.float32(im), contList, -1, (0,255,0))
    
	# Evaluamos la flecha si la hay
	if len(contList) > 0:
		# Calculo el centro de gravedad
		M = cv2.moments(contList[0])
		try:
			gravity_center = [M["m10"]/M["m00"], M["m01"] / M["m00"]]
		except:
			gravity_center = [M["m10"]/1, M["m01"]/1]
		
		#img_cont = cv2.circle(img_cont, (int(gravity_center[0]),int(gravity_center[1])), 2, (0,32,255), 2)
		        
		# Calculo el centro del rectangulo que contiene la flecha
		rect = cv2.minAreaRect(contList[0])
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		#im = cv2.drawContours(im,[box],0,(0,32,255),2)

		M = cv2.moments(box)
		try:
			center = [M["m10"]/M["m00"], M["m01"] / M["m00"]]
		except:
			center = [M["m10"]/1, M["m01"]/1]
        
		#img_cont = cv2.circle(img_cont, (int(center[0]),int(center[1])), 2, (0,255,255), 2)
        
		# Calculo la direccion en funcion de los centros calculados
		v_dir = [center[0]-gravity_center[0], center[1]-gravity_center[1]]
        
		if abs(v_dir[0]) > abs(v_dir[1]):
			if v_dir[0] > 0:
				acc[0] = acc[0] + 1
			else:
				acc[1] = acc[1] + 1
		else:
			if v_dir[1] >= 0:
				acc[2] = acc[2] + 1
			else:
				acc[3] = acc[3] + 1

		# Establezco la direccion de la flecha
		if acc[0] > acc[1] and acc[0] > acc[2] and acc[0] >= acc[3]:
			d = "izquierda"
			arrow = 0
		elif acc[1] >= acc[0] and acc[1] > acc[2] and acc[1] >= acc[3]:
			d = "derecha"
			arrow = 1
		elif acc[2] >= acc[0] and acc[2] >= acc[1] and acc[2] >= acc[3]:
			d = "frente"
			arrow = 2
		elif acc[3] > acc[0] and acc[3] > acc[1] and acc[3] > acc[2]:
			d = "atras"
			arrow = 3

	return img_cont, d, arrow, acc   


def eval_exit(img_cont, cent, arrow):

	if arrow == 0:
		exit = min(cent, key=itemgetter(0))
	elif arrow == 1:
		exit = max(cent, key=itemgetter(0))
	elif arrow == 2:
		exit = min(cent, key=itemgetter(1))
	elif arrow == 3:
		exit = max(cent, key=itemgetter(1))
    
	img_cont = cv2.circle(img_cont, (int(exit[0]),int(exit[1])), 8, (255,0,255), 15)
    
	if exit[1] < 60:
		eX = 60
	else:
		eX = exit[1]

	vel = round((151 - exit[1]) / 150, 2)
	ang = -round(((exit[0] - 160) * eX) / (320 * 25), 2)
	
	if vel > 0.7:
		vel = 0.7
	elif vel < 0.2:
		vel = 0.2	
	
	cv2.putText(img_cont, f"velocidad: {vel}", org_3, font, fontScale, color, thickness, cv2.LINE_AA)
	cv2.putText(img_cont, f"angulo: {ang}", org_4, font, fontScale, color, thickness, cv2.LINE_AA)
        
	return img_cont, vel, ang
	
def study_marks(img_cont, img_seg, acc, neigh):
    
	# Construyo imágenes a partir de las etiquetas de segmentación y extraigo los contornos
	img_grey_mark = (np.sum(img_seg==np.array([0,0,255]),axis=2)==3).astype("uint8")*255 
	contList,hier = cv2.findContours(img_grey_mark,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
	# Cojo el contorno mas grande
	if len(contList) != 0:
		contList = [max(contList, key = cv2.contourArea)]
		area = cv2.contourArea(contList[0])
        
		if area < 50:
			contList = []
            
	#img_cont = cv2.drawContours(np.float32(img_cont), contList, -1, (0,255,0))
    
	if len(contList) != 0:
		des =  orb_descriptors(contList[0], img_grey_mark)

		if des is not None:
			mark = neigh.predict([des])
			acc[int(mark)] = acc[int(mark)] + 1
            
			# Establezco la direccion de la flecha
			if acc[0] > acc[1] and acc[0] > acc[2] and acc[0] >= acc[3]:
				mark_text = "Hombre"
				print("Hombre")
			elif acc[1] >= acc[0] and acc[1] > acc[2] and acc[1] >= acc[3]:
				mark_text = "Escalera"
				print("Escalera")
			elif acc[2] >= acc[0] and acc[2] >= acc[1] and acc[2] >= acc[3]:
				mark_text = "Telefono"
				print("Telefono")
			elif acc[3] > acc[0] and acc[3] > acc[1] and acc[3] > acc[2]:
				mark_text = "Mujer"
				print("Mujer")
            
			text = cv2.putText(img_cont, f'marca: {mark_text}', org_5, font, fontScale, color, thickness, cv2.LINE_AA)
            
		else:
			acc = [0,0,0,0]
            
	return img_cont, acc


def orb_descriptors(cont, im):
    
    orb = cv2.ORB_create()
    ellip = cv2.fitEllipse(cont)
    cen, ejes, angulo = np.array(ellip[0]), np.array(ellip[1]), ellip[2]
    if angulo > 90:
        angulo -= 180
        
    kp = cv2.KeyPoint(cen[0], cen[1], np.mean(ejes)*1.3, angulo - 90)
    lkp, des = orb.compute(im, [kp])
    
    if des is not None:
        return np.unpackbits(des).T
    return None
