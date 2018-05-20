# IPython log file

import pandas as pd
s=pd.HDFStore('measLaserSpectra1526386349.h5')
s
s.keys()
for i in s.keys():
    if 'spectra' in i:
        d = s[i]
        intens = d.intens
        wl = d.wavelength
        bg = d.baseline
        plot(wl,intens,label=i)
        
legend()
NP1 = []
NP2 = []
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
        
s.keys()
ex_wl=[]
NP1=[]
NP={}
NP={'NP0':[],'NP1':[]}
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
        _,ex_wl,np = i.split('_')[-1]
        ex_wl = float(ex_wl)
        w = (wl>ex_wl/2-30)&(wl<(ex_wl/2+30))
        integr_SHG = intens[w].sum()
        NP[np].append([ex_wl,integr_SHG])
        
_
ex_wl
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
        ex_wl = float(ex_wl)
        w = (wl>ex_wl/2-30)&(wl<(ex_wl/2+30))
        integr_SHG = intens[w].sum()
        NP[np].append([ex_wl,integr_SHG])
        
NP
NP0 =array(NP['NP0'])
NP)
NP)
NP0
NP1 =array(NP['NP1'])
plot(NP0[:,0],NP0[:,1],'-')
plot(NP1[:,0],NP1[:,1],'--')
w
plot(wl,intens)
plot(wl,intens)
plot(wl[w],intens[w])
NP={'NP0':[],'NP1':[]}
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
        ex_wl = float(ex_wl)
        w = (wl>ex_wl/2-15)&(wl<(ex_wl/2+20))
        integr_SHG = intens[w].sum()
        NP[np].append([ex_wl,integr_SHG])
        
        
NP0 =array(NP['NP0'])
NP1 =array(NP['NP1'])
plot(NP0[:,0],NP0[:,1],'-')
plot(NP1[:,0],NP1[:,1],'--')
get_ipython().run_line_magic('logstart', '')
plot(NP1[:,0],NP1[:,1],'--',label='NP1')
plot(NP1[:-1,0],NP1[:-1,1],'--',label='NP1')
plot(NP0[:-1,0],NP0[:-1,1],'-',label='NP0')
legend()
xlabel('Ex, wavelength, nm')
ylabel('SHG')
s.keys()
s['/center_1000']
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
        if centers[index][1]>29: np = 'NP1'
        else: np = 'NP0'
        ex_wl = float(ex_wl)
        w = (wl>ex_wl/2-15)&(wl<(ex_wl/2+20))
        integr_SHG = intens[w].sum()
        NP[np].append([ex_wl,integr_SHG])
        
        
NP0 =array(NP['NP0'])
NP1 =array(NP['NP1'])
plot(NP0[:-1,0],NP0[:-1,1],'-',label='NP0')
centers[index][1]
centers[index]
.index
centers
centers
centers.sort()
centers.sort_values
centers.sort_values()
centers.sort_values(by=1)
centers.sort_values(by=0)
centers.sort_values(by=0)
NP=[]
NP=[[],[]]
NP
centers.sort_values(by=0)
centers.sort_values(by=0)[0]
centers.sort_values(by=0).index
centers.sort_values(by=0).index[0]
centers.sort_values(by=0).iloc[1]
centers.sort_values(by=0).iloc[0]
centers.sort_values(by=0).loc[0]
centers.sort_values(by=0).iloc[0]
centers.sort_values(by=0)
centers.sort_values(by=0).loc[0]
centers.sort_values(by=0).loc[0].index
centers.sort_values(by=0).loc[0]
centers.sort_values(by=0).loc[0]
centers.sort_values(by=0).iloc[0]
NP
ci = []
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
        cn=centers[index]
        else: np = 'NP0'
        ex_wl = float(ex_wl)
        w = (wl>ex_wl/2-15)&(wl<(ex_wl/2+20))
        integr_SHG = intens[w].sum()
        NP.append([cn[0],cn[1],ex_wl,integr_SHG])

        
        
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
        cn=centers[index]
        ex_wl = float(ex_wl)
        w = (wl>ex_wl/2-15)&(wl<(ex_wl/2+20))
        integr_SHG = intens[w].sum()
        NP.append([cn[0],cn[1],ex_wl,integr_SHG])

        
        
        
NP
NP=array(NP)
NP
NP[:,1]>29
w=NP[:,1]>29
NP0=NP[w]
NP0
NP1=NP[~w]
NP1
NP
plot(NP[:,1])
w=NP[:,1]>24
NP0=NP[w]
NP1=NP[~w]
plot(NP0[:-1,0],NP0[:-1,1],'-',label='NP0')
NP0=NP0[NP0[:,2].argsort()]
NP1=NP1[NP1[:,2].argsort()]
plot(NP0[:-1,0],NP0[:-1,1],'-',label='NP0')
plot(NP0[:-1,2],NP0[:,3],'-',label='NP0')
NP0
plot(NP0[:,2],NP0[:,3],'-',label='NP0')
plot(NP1[:,2],NP1[:,3],'--',label='NP1')
NP
plot(NP[:,0])
plot(NP[:,1])
plot(NP1[:,2],NP1[:,3],'--',label='NP1')
plot(NP1[:,2],NP1[:,3],'--',label='NP1')
plot(NP1[:,2],NP1[:,3],'--',label='NP0')
plot(NP1[:,2],NP1[:,3],'--',label='NP1')
plot(NP0[:,2],NP0[:,3],'-',label='NP0')
xlabel('Ex, wavelength, nm')
ylabel('SHG')
legend()
get_ipython().run_line_magic('cd', '../LN1_HWP45/')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cd', '../')
get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('cd', 'LN2_HWP45/')
get_ipython().run_line_magic('ls', '')
s.close()
