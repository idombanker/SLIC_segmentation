# Name: Banana's components detection
# Author: Joe
# Note: I reserve all the right for the final explanation

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pymeanshift as pms
from sklearn.svm import SVC
from sklearn.externals import joblib
import slic

class Joe_segmentation:

	def __init__(self, image, input_clf, number_regions=500, consistency=5):
		"""def __init__(self, image, number_regions=500, input_clf):"""
		self.original_image = cv2.imread(image)
		self.hsvimage = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
		self.number_regions = number_regions
		self.clf = joblib.load(input_clf)
		self.label_mask = np.array(slic.slic_n(self.original_image, number_regions, consistency),dtype=np.uint8)

		pass

	def label_color(self):
		out = np.ones(self.image.shape,dtype=np.uint8)
		for i in range(self.number_regions):
			newval = np.random.randint(256, size=3)
			out[self.label_mask==i] = newval
		return out
	
	def extract_mask(self,img,value):
		"""Help to extract the area selected by mouse or input area number"""

		# print("Extracting mask...")
		temp_mask = np.copy(img)
		temp_mask = temp_mask + 1
		temp_mask[temp_mask!=(value+1)]=0
		temp_mask[temp_mask>0]=1
		return temp_mask

	#	this is the method to define a mouse callback function. Several events are given in OpenCV documentation
	def my_mouse_callback(self,event,x,y,flags,param):

		if event==cv2.EVENT_LBUTTONDOWN:		# here event is left mouse button double-clicked
			print "Position(%d,%d):"%(x,y)
			print "%d"%label_mask[y][x]
			mask = extract_mask(label_mask,label_mask[y][x])
			output = self.original_image * cv2.merge([mask,mask,mask])
			testdata = feature_calc_save(self.hsvimage,mask)
			results = clf.predict([testdata])

			print " svm_result:"
			print results[0]
			feature_show(self.hsvimage,mask)
		

			# cv2.imshow("window_%d"%label_mask[y][x],output)
		# elif event==cv2.EVENT_RBUTTONDOWN:
		# 	print "Position(%d,%d):"%(x,y)
		# 	print "%d"%im[y][x][0]
		# 	mask = extract_mask(im,im[y][x][0])
		# 	output = o_im * mask
		# 	cv2.imshow("window_%d"%im[y][x][0],output)
		# 	feature_calc_save(im,mask)
			# print "Right Click Show"

			#text="{0},{1}".format(x,y)
			#cv.PutText(im,text,(x+5,y+5),f,cv.RGB(0,255,255))




	# save the feature in two adnarray ( flesh and skin ) 
	def feature_calc_save(self, im, roi_mask):

		histh = cv2.calcHist([im],[0],roi_mask,[15],[0,180])
		hists = cv2.calcHist([im],[1],roi_mask,[20],[0,256])
		histv = cv2.calcHist([im],[2],roi_mask,[20],[0,256])

		cv2.normalize(histh,histh,0,1,cv2.NORM_MINMAX)
		cv2.normalize(hists,histv,0,1,cv2.NORM_MINMAX)
		cv2.normalize(histv,hists,0,1,cv2.NORM_MINMAX)

		feature_vector = np.append(histh,hists)
		feature_vector = np.append(feature_vector,histv)


		return feature_vector


	def feature_show(self, im, roi_mask):
		'''input: rgb image and mask
		   output: histogram of the roi area 
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


	def test_click(self):
		"""test_click(o_im,label_mask,number_regions)"""
		output_mask = self.label_mask.copy()
		for i in range(self.number_regions):
			mask = extract_mask(self.label_mask,i)
			testdata = feature_calc_save(self.hsvimage,mask)
			results= self.clf.predict([testdata])
			output_mask[np.where(self.label_mask == i)] = results[0]
			print results


		output = [1,2,3]

		for i in range(3):
			out_put = np.zeros(self.label_mask.shape)
			out_put[np.where(output_mask == i)] = 255
			output[i] = out_put
		out = cv2.merge([output[0],output[1],output[2]])

		while(1):
			cv2.namedWindow("Display",1)
			cv2.setMouseCallback("Display",my_mouse_callback,self.segment_result)	#binds the screen,function and image
			cv2.imshow("Display",self.segment_result)
			cv2.imshow("out",out)
			cv2.imshow("original",self.original_image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			if cv2.waitKey(15)%0x100==27:
			    cv2.destroyAllWindows()
			    break	



	def test(self):
		output_mask = self.label_mask.copy()

		for i in range(self.number_regions):
			mask = self.extract_mask(self.label_mask, i)
			testdata = self.feature_calc_save(self.hsvimage,mask)
			results= self.clf.predict([testdata])
			output_mask[np.where(self.label_mask == i)] = results[0]

		output = [1,2,3]

		for i in range(3):
			out_put = np.zeros(self.label_mask.shape)
			out_put[np.where(output_mask == i)] = 255
			output[i] = out_put

		out = cv2.merge([output[0],output[1],output[2]])
		cv2.namedWindow("out_put",cv2.WINDOW_AUTOSIZE)
		cv2.namedWindow("original",cv2.WINDOW_AUTOSIZE)
		cv2.imshow("out_put",out)
		cv2.imshow("original",self.original_image)
		cv2.imwrite("svm_output.jpg",out)
		cv2.waitKey(0)
		cv2.destroyAllWindows()



# flesh = 0 skin = 1 other =2
def svm_training(sk_feature, fl_feature, ot_feature,data_size=150):
	"""generate clf of svm"""

	total_size = data_size * 3

	sk = np.load(sk_feature)
	fl = np.load(fl_feature)
	ot = np.load(ot_feature)
	temp_sk = range(data_size)
	temp_fl = range(data_size)
	temp_ot = range(data_size)
	out = range(total_size)
	for i in temp_sk:
		temp_sk[i] = sk[i]


	for i in temp_fl:
		temp_fl[i] = fl[i]

	for i in temp_ot:
		temp_ot[i] = ot[i]

	temp_sk = np.array(temp_sk)
	temp_fl = np.array(temp_fl)
	temp_ot = np.array(temp_ot)

	sk_label = np.ones((data_size,),dtype = np.int)
	fl_label = np.ones((data_size,),dtype = np.int)-1
	ot_label = np.ones((data_size,),dtype = np.int)+1

	# flesh = 0 skin = 1 other =2

	index = np.append([sk_label],[fl_label])
	index = np.append(index,[ot_label])

	# combine above label together we have label_axis for svm

	X = np.append([temp_sk],[temp_fl],axis=1)
	X = np.append([X[0]],[temp_ot],axis=1)
	X = X[0]

	# append above together we have feature axis for svm

	X = np.array(X)

	clf = SVC(C=1,kernel='rbf',gamma='auto',probability=True,decision_function_shape='ovo')
	clf.fit(X,index)
	joblib.dump(clf, './svmfile.pkl')
	return clf

if __name__ == "__main__":
	c = svm_training('./outfile_sk.npy', './outfile_fl.npy', './outfile_ot.npy')
	a = Joe_segmentation('banana.jpg', './svmfile.pkl')
	a.test()














