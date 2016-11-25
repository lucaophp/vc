import matplotlib.pyplot as plt 
import numpy as np 
from scipy.ndimage import filters,measurements
from skimage import feature, measure, data, color, exposure, morphology,util, filter
from scipy import misc
from sklearn import cluster,neighbors
#carregar imagem
im=misc.imread("objetos.bmp")
#filtro
im=filters.median_filter(im,5)
#segmentacao
th=filter.threshold_otsu(im)
im_b=im<th
ee=np.ones((5,5))
#im_b=morphology.binary_closing(im_b,ee)
#im_b=morphology.binary_opening(im_b,ee)

im_b=morphology.binary_erosion(im_b,ee)
im_b=morphology.binary_closing(im_b,ee)
im_b=morphology.binary_dilation(im_b,ee)
im_b=morphology.remove_small_objects(im_b, 40)

#label
im_r, num_obj = measurements.label(im_b)
print num_obj
#extracao de caracteristicas
props=measure.regionprops(im_r,im_b)
for i,p in enumerate(props):
	print "Area: %.2f"%(p.area)
	print "Perimetro: %.2f"%(p.perimeter)
	# Constroi ndarray do objeto i
	F_i = np.array( [p.area, (p.centroid)[0],
	(p.centroid)[1], p.convex_area,
	p.eccentricity, p.equivalent_diameter,
	p.euler_number, p.extent,
	p.filled_area, p.major_axis_length,
	p.max_intensity, p.mean_intensity,
	p.min_intensity, p.minor_axis_length,
	p.orientation, p.perimeter,
	p.solidity] )
	# TESTE
	print 'Vetor de caracteristicas do objeto: ', i
	print F_i
	# Justa os vetores e carac. na matriz de caract. F
	if i==0:
		F = F_i
	else:
		F = np.vstack((F,F_i))
#KMEANS para rotular entre porcas e parafusos
km=cluster.KMeans(2)

c=km.fit(F)
labels= c.labels_
#grava no csv
np.savetxt('Teste_F.csv', F, delimiter=' , ', fmt='%.4f')
#classifica usando KNN
knn=neighbors.KNeighborsClassifier(1)
knn.fit(F[:3:],labels[:3:])
predito=knn.predict(F[3::])
res=[]
lfat=labels[3::].copy()


print predito
print lfat
for i,j in enumerate(predito):
	
	res.append(predito[i]==lfat[i])
print res


#PLOTA A FIGURA DE ROTULOS
plt.figure()
plt.imshow(im_r,cmap='gray')
plt.show()