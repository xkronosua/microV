from pylab import *
import pandas as pd
from lmfit.models import GaussianModel
from glob import glob
from scipy import signal
from peakutils.peak import indexes
from scipy.signal import medfilt
from lmfit.models import GaussianModel, LorentzianModel
import traceback
import numpy as np

with pd.HDFStore(glob('*.h5')[0]) as store:
	l = store.keys()
	l.sort(key=lambda x: int(x.split('_')[-1]))
	spectra = []
	for i in l:
		if 'spectra' in i:
			spectra.append([store[i],int(i.split('_')[1])])

	THG = []
	SHG = []
	Peak2 = []
	ex_wl = []
	ax = []
	fig1 = figure(1)
	fig2 = figure(2)
	ax2_1 = fig2.add_subplot(2,1,1)
	ax2_2 = fig2.add_subplot(2,1,2)
	fig3 = figure(3)
	ax3 = fig3.add_subplot(1,1,1)
	out = []
	for i,d in enumerate(spectra):
		intens = d[0]['intens'].values
		wavelength = d[0]['wavelength'].values
		w = wavelength<330
		intens = medfilt(intens - intens[w].mean(), 7)
		ex_wavelength = d[1]

		max_peakind = indexes(intens, thres=5.0/max(intens), min_dist=15)
		if len(max_peakind)>3:
			max_peakind = indexes(intens, thres=17.0/max(intens), min_dist=15)
		if len(max_peakind)>0:
			print(ex_wavelength, wavelength[max_peakind])
			ax.append( fig1.add_subplot(6,5,i+1))
			ax[-1].plot(wavelength,intens,'b')
			ax[-1].plot(wavelength[max_peakind],intens[max_peakind],'ro')
			gauss = []
			gauss.append( GaussianModel(prefix='g1_'))
			pars = gauss[0].guess(intens,x=wavelength)

			mod = gauss[-1]
			for j in range(1,len(max_peakind)+1):
					if j!=1:
						gauss.append( GaussianModel(prefix="g"+str(j)+"_"))
						pars.update(gauss[-1].make_params())
					pars["g"+str(j)+"_center"].set(wavelength[max_peakind[j-1]],
						min=wavelength[max_peakind[j-1]]-10, max=wavelength[max_peakind[j-1]]+10)
					pars["g"+str(j)+"_amplitude"].set(intens[max_peakind[j-1]]*10, min=intens.min(), max=intens.max()*20)

					pars["g"+str(j)+"_sigma"].set(5,min=2,max=20)
					if j!=1:
						mod += gauss[-1]


			init = mod.eval(pars, x=wavelength)
			out = mod.fit(intens, pars, x=wavelength)

			#ax[-1].plot(wavelength,init,'g')
			ax[-1].plot(wavelength,out.best_fit,'r')
			ax[-1].set_title('@'+str(ex_wavelength))
		try:
			components = out.components
			if len(components)==3:
				p1_c, p2_c, p3_c = out.params['g1_center'].value, out.params['g2_center'].value, out.params['g3_center'].value
				area1, area2, area3 =  out.params['g1_amplitude'].value, out.params['g2_amplitude'].value, out.params['g3_amplitude'].value
				peak = np.array([p1_c, p2_c, p3_c])
				area = np.array([area1,area2,area3])
				ax2_1.plot(ex_wavelength,p2_c,'o')
				ax2_1.set_ylabel('Peak2 center, nm')
				ax2_2.plot(ex_wavelength,area2,'o')
				ax2_2.set_xlabel('Ex. wavelength, nm')
				ax2_2.set_ylabel('Peak2 area')
				THG.append(area1)
				SHG.append(area3)
				Peak2.append([ex_wavelength, p2_c, area2])
			if len(components)==2:
				p1_c, p3_c = out.params['g1_center'].value, out.params['g2_center'].value
				area1, area3 =  out.params['g1_amplitude'].value, out.params['g2_amplitude'].value
				THG.append(area1)
				SHG.append(area3)
			if len(components)==1:
				p1_c = out.params['g1_center'].value
				area1 =  out.params['g1_amplitude'].value
				THG.append(area1)
				SHG.append(0)
			if len(max_peakind)==0:
				THG.append(0)
				SHG.append(0)
			ax3,plot(wavelength,intens,label="@"+str(ex_wavelength))
			ex_wl.append(ex_wavelength)
		except:
			traceback.print_exc()
	ax3,legend()
	THG = np.array(THG)
	SHG = np.array(SHG)
	ex_wavelength = np.array(ex_wl)
