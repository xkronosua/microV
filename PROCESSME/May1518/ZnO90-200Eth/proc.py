import pandas as pd
from pylab import *

ranges=[[0,82],[82,84],[84,87],[87,90]]
with pd.HDFStore('measLaserSpectra1526398705.h5') as s:
	NP=[]
	for i in s.keys():
		if 'spectra' in i:
			if 'NP0' in i:
				d = s[i]
				intens = d.intens
				wl = d.wavelength
				bg = d.baseline
				plot(wl,intens,'-',label=i)
			else:
				d = s[i]
				intens = d.intens
				wl = d.wavelength
				bg = d.baseline
				plot(wl,intens,'--',label=i)

			txt,ex_wl,np = i.split('_')
			centers = s['/center_'+ex_wl]
			index = int(np[-1])
			try:
				cn=centers.loc[index]
			except: continue
			ex_wl = float(ex_wl)
			w = (wl>ex_wl/2-15)&(wl<(ex_wl/2+20))
			integr_SHG = intens[w].sum()
			w3 = (wl>ex_wl/3-15)&(wl<(ex_wl/3+20))
			integr_THG = intens[w3].sum()
			bl = (bg-bg.min())
			bl = bl[w3].sum()
			try:
				NP.append([cn[0],cn[1],ex_wl,integr_SHG,integr_THG,bl])
			except:
				pass


NP=array(NP)
k=(NP[:,0]**2+NP[:,1]**2)**0.5
NP_l = []
for w in ranges:
	ww = (k>w[0])&(k<w[1])
	n = NP[ww]
	NP_l.append(n)

NP_l = []
for w in ranges:
	ww = (k>w[0])&(k<w[1])
	n = NP[ww]
	NP_l.append(n)

figure(2)

for j,n in enumerate(NP_l):
	plot(n[:,2],n[:,3],label='NP'+str(j))


legend()
xlabel('Ex, wavelength, nm')
ylabel('SHG')
savefig('ZnO_SHG.png')
figure(3)

for j,n in enumerate(NP_l):
	plot(n[:,2],n[:,3]/n[:,5],label='NP'+str(j))


legend()
xlabel('Ex, wavelength, nm')
ylabel('SHG/interf_THG')
savefig('ZnO_SHG_norm.png')

figure(4)


for j,n in enumerate(NP_l):
	plot(n[:,2],n[:,4],label='NP'+str(j))



xlabel('Ex, wavelength, nm')
ylabel('THG')
legend()
savefig('ZnO_THG.png')
figure(5)

for j,n in enumerate(NP_l):
	plot(n[:,2],n[:,4]/n[:,5],label='NP'+str(j))



xlabel('Ex, wavelength, nm')
ylabel('THG/interf_THG')
legend()

savefig('ZnO_THG_norm.png')
