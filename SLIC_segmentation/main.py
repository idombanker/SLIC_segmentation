# -*- coding:utf-8 -*-

#Name: Banana's components detection
# Author: Joe
# Note: I reserve all the right for the final explanation

import cv2
import numpy as np
import slic

from sklearn.svm import SVC
from sklearn.externals import joblib



original = cv2.imread('./banana.jpg')

o_im = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

# testing

number_regions = 500
region_labels = slic.slic_n(original, 500, 5)

label_mask = np.array(region_labels.copy(),dtype=np.uint8)

clf = joblib.load("banana_clf.pkl")

def label_color(label, number_regions, shape):

	out = np.ones(shape,dtype=np.uint8)
	for i in range(number_regions):
		newval = np.random.randint(256, size=3)

		out[label==i] = newval
	return out

segment_result = label_color(region_labels,number_regions,original.shape )
	
def extract_mask(img,value):
	# print("Extracting mask...")
	temp_mask = np.copy(img)
	# print temp_mask
	temp_mask = temp_mask + 1
	temp_mask[temp_mask!=(value+1)]=0
	temp_mask[temp_mask>0]=1


	# print("Extracting ")
	return temp_mask

#	this is the method to define a mouse callback function. Several events are given in OpenCV documentation
def my_mouse_callback(event,x,y,flags,param):
	if event==cv2.EVENT_LBUTTONDOWN:		# here event is left mouse button double-clicked
		print "Position(%d,%d):"%(x,y)
		print "%d"%label_mask[y][x]
		mask = extract_mask(label_mask,label_mask[y][x])
		output = original * cv2.merge([mask,mask,mask])
		testdata = feature_calc_save(o_im,mask)
		results = clf.predict([testdata])

		print " svm_result:"
		print results[0]
		feature_show(o_im,mask)
	

# save the feature in two adnarray ( flesh and skin ) 
def feature_calc_save(im,roi_mask):

	histh = cv2.calcHist([im],[0],roi_mask,[15],[0,180])
	hists = cv2.calcHist([im],[1],roi_mask,[20],[0,256])
	histv = cv2.calcHist([im],[2],roi_mask,[20],[0,256])

	cv2.normalize(histh,histh,0,1,cv2.NORM_MINMAX)
	cv2.normalize(hists,histv,0,1,cv2.NORM_MINMAX)
	cv2.normalize(histv,hists,0,1,cv2.NORM_MINMAX)

	feature_vector = np.append(histh,hists)
	feature_vector = np.append(feature_vector,histv)


	return feature_vector


def feature_show(im,roi_mask):
	'''input:rgb image and mask
	   output: 
	''' 
	histh = cv2.calcHist([im],[0],roi_mask,[15],[0,180])
	hists = cv2.calcHist([im],[1],roi_mask,[20],[0,256])
	histv = cv2.calcHist([im],[2],roi_mask,[20],[0,256])

	cv2.normalize(histh,histh,0,1,cv2.NORM_MINMAX)
	cv2.normalize(hists,hists,0,1,cv2.NORM_MINMAX)
	cv2.normalize(histv,histv,0,1,cv2.NORM_MINMAX)
	plt.figure()
	plt.text(17,1,"h:red\ns:green\ni:blue")
	plt.plot(histh,color = 'r')
	plt.plot(histv,color = 'g')
	plt.plot(hists,color = 'b')
	plt.xlim([0,20])
	plt.show()

	feature_vector = np.append(histh,hists)
	feature_vector = np.append(feature_vector,histv)


	return feature_vector


# flesh = 0 skin = 1 other =2




def test_click():

	output_mask = label_mask.copy()
	for i in range(number_regions):
		print i
		print ":"
		mask = extract_mask(label_mask,i)
		# print "number of region is :"
		testdata = feature_calc_save(o_im,mask)
		results= clf.predict([testdata])
		output_mask[np.where(label_mask == i)] = results[0]
		print results


	output = [1,2,3]

	for i in range(3):
		out_put = np.zeros(label_mask.shape)
		out_put[np.where(output_mask == i)] = 255
		output[i] = out_put
	out = cv2.merge([output[0],output[1],output[2]])

	while(1):
		cv2.namedWindow("Display",1)
		cv2.setMouseCallback("Display",my_mouse_callback,segment_result)	#binds the screen,function and image
		cv2.imshow("Display",segment_result)
		cv2.imshow("out",out)
		cv2.imshow("original",original)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		if cv2.waitKey(15)%0x100==27:
		    cv2.destroyAllWindows()
		    break	



def test():
	output_mask = label_mask.copy()
	for i in range(number_regions):
		print i
		print ":"
		mask = extract_mask(label_mask,i)
		# print "number of region is :"
		testdata = feature_calc_save(o_im,mask)
		results= clf.predict([testdata])
		output_mask[np.where(label_mask == i)] = results[0]
		# print results


	output = [1,2,3]

	for i in range(3):
		out_put = np.zeros(label_mask.shape)
		out_put[np.where(output_mask == i)] = 255
		output[i] = out_put
	# print out_put
	out = cv2.merge([output[0],output[1],output[2]])
	cv2.imshow("out_put",out)
	cv2.imshow("original",original)
	cv2.imwrite("svm_output.jpg",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

test_click()

