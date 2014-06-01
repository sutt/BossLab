import sklearn
import numpy as np
from scipy import ndimage

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import skimage
from skimage import io
from skimage.filter import threshold_otsu

from skimage.feature import peak_local_max
from skimage.morphology import watershed

class Plotting:

	@staticmethod
	def algo1(inp_img):
		labels = Finding.img_to_labels1(inp_img)
		out_fig = Plotting.make_windows(labels,inp_img)
		return out_fig
	
	@staticmethod
	def boundbox1(label):
	
	#     np.nonzero(np.apply_over_axes( np.argmax, thelabel,[1]))[0].min()
		minX = np.nonzero(np.apply_over_axes(np.argmax,(label),[1]))[0].min()
		maxX = np.nonzero(np.apply_over_axes(np.argmax,(label),[1]))[0].max()
		minY = np.nonzero(np.apply_over_axes(np.argmax,(label),[0]))[1].min()
		maxY = np.nonzero(np.apply_over_axes(np.argmax,(label),[0]))[1].max()
		return [minX,maxX,minY,maxY]

	
	@staticmethod
	def make_windows(inp_labels, inp_img):
		bb0 = []
		for i in range(1,inp_labels.max() + 1):
			label_i = (inp_labels == i)
			bb0.append(Plotting.boundbox1(label_i))

		rect_arr = [Plotting.buildrect(bb0[i]) for i in np.arange(bb0.__len__())]

		fig = None
		s0 = None
		fig, (s0) = plt.subplots(nrows = 1, ncols = 1, figsize = (12,12))
		s0.imshow(inp_img, cmap='gray')
		for rect in rect_arr:
			s0.add_patch(rect)
		fig.show()
		return fig
		
		
	
	@staticmethod
	def buildrect(arr):
		
		minY,maxY,minX,maxX = arr
		
		rect1 = mpatches.Rectangle(xy=(minX,minY)
							   ,width = maxX - minX
							   ,height = maxY - minY
							   ,fill = False
							   ,edgecolor = 'red'
							   ,linewidth = 2)
		return rect1


class Finding:


	@staticmethod
	def img_to_labels1(inp_img):
		""" Takes input img, 
			Returns labels from watershed method
		"""
		thresh = threshold_otsu(inp_img)
		otsu0 = inp_img < thresh
		dist0 = ndimage.distance_transform_edt(otsu0)
		j=34
		peaks0 = peak_local_max(dist0 ,indices = False,footprint=np.ones((j,j))) 
		
		marker0 = ndimage.label(peaks0)[0]
		labels0 = watershed(-dist0, marker0, mask=otsu0)
		
	#     plt.imshow(labels0)
	#     plt.show()
		return labels0

	