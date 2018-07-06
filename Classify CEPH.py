# This file loads every raw light curve of which name starts with "OGLE", performs sigma clipping on each light curve,
# and classifies its variability. 
CLASS = 'CEPH'
# Import modules
import numpy as np
import glob
from time import time
import csv
# - - - A E S T H E T I C S - - -
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style()
# - - - - - - - - - - - - - - - - 

t0 = time()
import upsilon
# Load a classification model.
rf_model = upsilon.load_rf_model()
print "Upsilon and RF model is loaded in", time()-t0, 'seconds. Starting to load catalogue and files.' 

StarName = []
Label_pred =[]
Label_true =[]
P_true = []
P_pred = []
varmode =[]
ID = []
catperiod = []
Tmax = []
probz =[]

# - - - L O A D - C A T A L O G U E - - -
funda = sorted(glob.glob('cepF.*'))
for name in funda:
	pf = np.genfromtxt(fname=name, delimiter=',', usecols=(3))
	IDfs = np.genfromtxt(fname=name, delimiter=',', dtype=np.str, usecols=(0))
	pf = np.nan_to_num(pf)

	pfs = [x for x in pf]
	IDf = [z for z in IDfs]
	
	catperiod.extend(pfs)
	ID.extend(IDf)

	vars = ['F' for x in pf]
	varmode.extend(vars)

first = sorted(glob.glob('cep1O.*'))
for name in first:
	pf = np.genfromtxt(fname=name, delimiter=',', usecols=(3))
	IDfs = np.genfromtxt(fname=name, delimiter=',', dtype=np.str, usecols=(0))
	pf = np.nan_to_num(pf)
	
	pfs = [x for x in pf]
	IDf = [z for z in IDfs]
	catperiod.extend(pfs)
	ID.extend(IDf)

	vars = ['1O' for x in pf]
	varmode.extend(vars)

other = sorted(glob.glob('cep*'))
other.remove('cepF.dat')
other.remove('cep1O.dat')

for name in other:
	pf = np.genfromtxt(fname=name, delimiter=',', usecols=(3))
	IDfs = np.genfromtxt(fname=name, delimiter=',', dtype=np.str, usecols=(0))
	pf = np.nan_to_num(pf)
	
	pfs = [x for x in pf]
	IDf = [z for z in IDfs]
	catperiod.extend(pfs)
	ID.extend(IDf)

	vars = ['Other' for x in pf]
	varmode.extend(vars)

#print ID, catperiod, varmode

# - - - L O A D - F I L E - - -
filenames = sorted(glob.glob('OGLE-LMC-CEP*'))

# LABEL EACH CLASSIFICATION
urutan=1

for f in filenames:
	t1=time()

	starID = f[:-4]
	t, m, e = np.loadtxt(fname=f, unpack=True)

	bindices_zero = (m < 0)

	# CLEAN SPURIOUS PHOTOMETRY VALUES and NaNs
	#print "The NaNs are in mag", np.argwhere(np.isnan(m)), "or date ", np.argwhere(np.isnan(t)), "or error ", np.argwhere(np.isnan(e))
	nawal = len(m)
	indices_zero = np.arange(len(m))[bindices_zero]

	m = np.delete(m, indices_zero)
	t = np.delete(t, indices_zero)
	e = np.delete(e, indices_zero)
	#print "Omitted data (spurious values) = ", nawal-len(m)
	m = np.nan_to_num(m)
	t = np.nan_to_num(t)
	e = np.nan_to_num(e)

	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

	bfores = len(m)
	# Perform sigma-clipping
	t, m, e = upsilon.utils.sigma_clipping(t, m, e, threshold=3, iteration=1)
	print "Removed in sigma-clipping: ", len(m)-bfores

	if len(m)>1:
		# Extract the fatures and label the light curve.
		start = "#" + str(urutan) + " --- CLASSIFICATION OF " + starID + " HAS BEEN INITIALIZED.---"
		print start

		#Catalogue indexing
		wherestar = ID.index(starID)

		# Extract Features
		
		e_features = upsilon.ExtractFeatures(t, m, e)
		t2 = time()
		e_features.run()
		features = e_features.get_features()
		print "Features are extracted in ", time()-t2, "seconds."

		mean_m = np.mean(m)
		std = np.std(m)
		
		#print 'The period of', f, '=', features['period']
		P_pred.append(features['period'])

		#Create an array of running mean by convolution of 10-nearest points
		#N = 20
	   	#m_convolve = np.convolve(m, np.ones((N,))/N, mode='valid')


		# Classify the light curve
		label, probability, flag = upsilon.predict(rf_model, features)
		
		#print "This is suspected as ", label
		if flag==1:

			#Identify star and pick period from catalogue if it's identified as a non-variable
			
			features['period'] = catperiod[wherestar]
			print "New period is ", features['period'] 
			#Classify ONCE again.
			labeln, probabilityn, flag = upsilon.predict(rf_model, features)

			# RECLASSIFY AFTER DOUBLING PERIOD, CHECK IF PROBABILITY IS HIGHER.	
			
			
			print f, 'is successfully RECLASSIFIED as', labeln, 'with the class probability', probabilityn, 'in', time()-t1, 'seconds. Period was taken from catalogue and probability is now higher.'	
			print " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
			probability = probabilityn
			label = labeln

		else:
			print f, 'is successfully classified as', label, 'with the class probability', probability, 'in', time()-t1, 'seconds'
			print " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
			 				
		
		#Make A Phase-Folded Light Curve
		P = features['period']
		fase_ = (t-t[0])/(2*P)
		fase = (fase_) - np.floor(fase_)
		fase = fase - fase[np.argmax(m)]
		fase_negatif = fase[fase < 0.]
		fase = fase.tolist()
		arg_fase_negatif = []
		for neg in fase_negatif:
			# arg_fase_negatif.append(fase.index(neg))
			index = fase.index(neg)
			fase[index] = 1 + neg	
		fase = [phase*2 for phase in fase]

		plt.figure(urutan)
		plt.title("%s | P = %f | mag = %f \n | VAR = %s probability = %f" %(starID, features['period'], mean_m, label, probability))
		plt.xlabel('Phase')   
		plt.ylabel('I-band Magnitude')
		plt.errorbar(fase,m,yerr=e,linestyle='none',marker='o',ms=6,
		 elinewidth=1, mfc='#34495f', ecolor='#95a5a6', capsize=30) 

		#Invert y-axis and save figure.
		plt.gca().invert_yaxis()
		plt.savefig("%f %s - #%d  - %s.png" %(probability, label, urutan, starID), dpi=150)
		
		#Load Catalogue Classification and period
		catstring = 'CEPH_'
		P_cat = catperiod[wherestar]
		thetrue = catstring + varmode[wherestar]
		
		

		#Save classification and period
		StarName.append(starID)
		Label_pred.append(label)
		Label_true.append(thetrue)
		P_true.append(catperiod[wherestar])
		probz.append(probability)
		
		urutan = urutan+1
		if urutan>50:
			break
	else:
		print "Data points are too few. Skipped to the next star."
		print " - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"

import pandas as pd
df = pd.DataFrame({'#Label_pred' : Label_pred, 'Star ID' : StarName, 'Label_true' : Label_true, 'P_true': P_true, 'P_extracted': P_pred})    
df.to_csv('result %s.csv' %(CLASS), index=False, encoding='utf-8')
print "Result is saved as result CEPH.csv"

cnf_matrix = pd.crosstab(Label_true, Label_pred, rownames=['True'], colnames=['Predicted'], margins=True)
cnf_matrix.to_csv('CF - %s.csv' %(CLASS), index=True, encoding='utf-8')
print ('Confusion Matrix is saved as CF -.csv' %(CLASS))
'''
from pandas_ml import ConfusionMatrix
cnf_matrix_pic = ConfusionMatrix(Label_true, Label_pred)
cnf_matrix_pic.plot(normalized=True)
plt.savefig("Confusion Matrix of %s.png" %(CLASS), dpi=200)
'''