# IPython log file

import pandas as pd
s=pd.HDFStore('measLaserSpectra1527004871.h5')
s
s.keys()
k=s.keys()
for i in k:
    print(i[1:8])
for i in k:
    if i[1:8] == 'spectra':
        l.append(i)
        
l=[]
for i in k:
    if i[1:8] == 'spectra':
        l.append(i)
        
l
s.get_storer('/trackB_1240').attrs
s.get_storer(l[0]).attrs
s.get_storer(l[0])
s[l[0]]
s[l[0]].attrs
l
ex_wl = arange(950,1180,5)
ex_wl
data=[]
for wl in ex_wl:
    d = []
    interf = s['/spectra_'+str(wl)+'_interface']
    m = interf.max()
    for i in range(4):
        try:
            tmp = s['/spectra_'+str(wl)+'_NP'+str(i)]
            w = (tmp.index>wl//3-20)&(tmp.index<wl//3+20)
            THG = tmp.intens[w].sum()/m
            d.append(THG)
        except:
            d.append(0)
    data.append(d)
   
data
data[0]
data[0][0]
data = array(data)
data
data=[]
interf
interf.max()
interf.max().value()
interf.max().value
interf.max()
data
for wl in ex_wl:
    d = []
    interf = s['/spectra_'+str(wl)+'_interface']
    m = interf.max()
    for i in range(4):
        try:
            tmp = s['/spectra_'+str(wl)+'_NP'+str(i)]
            w = (tmp.index>wl//3-20)&(tmp.index<wl//3+20)
            THG = float(tmp.intens[w].sum()/m)
            d.append(THG)
        except:
            d.append(0)
    data.append(d)
   
data
data=array(data)
data
plot(ex_wl,data[:,0])
plot(ex_wl,data[:,1])
plot(ex_wl,data[:,2])
plot(ex_wl,data[:,3])
plot(ex_wl,data[:,4])
data=[]
for wl in ex_wl:
    d = []
    interf = s['/spectra_'+str(wl)+'_interface']
    w = (interf.index>wl//3-20)&(interf.index<wl//3+20)
    m = interf[w].sum()
    for i in range(4):
        try:
            tmp = s['/spectra_'+str(wl)+'_NP'+str(i)]
            w = (tmp.index>wl//3-20)&(tmp.index<wl//3+20)
            THG = float(tmp.intens[w].sum()/m)
            d.append(THG)
        except:
            d.append(0)
    data.append(d)
   
data
data=array(data)
plot(ex_wl,data[:,1])
plot(ex_wl,data[:,0])
plot(ex_wl,data[:,2])
plot(ex_wl,data[:,3])
data=[]
for wl in ex_wl:
    d = []
    interf = s['/spectra_'+str(wl)+'_interface']
    expos = s.get_storer('/spectra_'+str(wl)+'_interface').attrs.exposure
    w = (interf.index>wl//3-20)&(interf.index<wl//3+20)
    m = interf[w].sum()/expos
    for i in range(4):
        try:
            tmp = s['/spectra_'+str(wl)+'_NP'+str(i)]
            w = (tmp.index>wl//3-20)&(tmp.index<wl//3+20)
            THG = float(tmp.intens[w].sum()/m)/expos
            d.append(THG)
        except:
            d.append(0)
    data.append(d)
   
for wl in ex_wl:
    d = []
    interf = s['/spectra_'+str(wl)+'_interface']
    expos = s.get_storer('/spectra_'+str(wl)+'_interface').attrs.exposure
    w = (interf.index>wl//3-20)&(interf.index<wl//3+20)
    m = interf[w].sum()
    for i in range(4):
        try:
            tmp = s['/spectra_'+str(wl)+'_NP'+str(i)]
            w = (tmp.index>wl//3-20)&(tmp.index<wl//3+20)
            THG = float(tmp.intens[w].sum()/m)
            d.append(THG)
        except:
            d.append(0)
    data.append(d)
   
data=[]
for wl in ex_wl:
    d = []
    interf = s['/spectra_'+str(wl)+'_interface']
    #expos = s.get_storer('/spectra_'+str(wl)+'_interface').attrs.exposure
    w = (interf.index>wl//3-20)&(interf.index<wl//3+20)
    m = interf[w].sum()
    for i in range(4):
        try:
            tmp = s['/spectra_'+str(wl)+'_NP'+str(i)]
            w = (tmp.index>wl//3-20)&(tmp.index<wl//3+20)
            THG = float(tmp.intens[w].sum()/m)
            d.append(THG)
        except:
            d.append(0)
    data.append(d)
   
data=array(data)
plot(ex_wl,data[:,0])
plot(ex_wl,data[:,1])
plot(ex_wl,data[:,2])
plot(ex_wl,data[:,3])
xlabel('Ex, wavelength, nm')
ylabel('THG')
title('ZnO20nmEth')
ylabel('THG/interfTHG')
get_ipython().run_line_magic('pwd', '')
savefig('ZnO20nmEth')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('logstart', '')
