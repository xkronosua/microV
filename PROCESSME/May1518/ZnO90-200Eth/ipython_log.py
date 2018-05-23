# IPython log file

import pandas as pd
s=pd.HDFStore('measLaserSpectra1526398705.h5')
s
s.keys()
ex_range=np.arange(1000,1240,5)
ex_range
ex_range=np.arange(1000,1245,5)
ex_range=np.arange(1000,1245,5)
ex_range
for ewl in ex_range:
    sp=[]
    for i in range(3):
        print(i)
        
for ewl in ex_range:
    sp=[]
    for i in range(4):
        spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)
        plot(spectra.index,spectra.intens)
     
        
for ewl in ex_range:
    sp=[]
    for i in range(4):
        spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
        plot(spectra.index,spectra.intens)
     
        
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            plot(spectra.index,spectra.intens)
        except: pass
        
len(ex_wl)
len(ex_range)
ax=[]
ax=[subplot(7,8,i+1) for i in range(49)]
n=0
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
        except: pass
    n=n+1
   
legend()
ax=[subplot(7,8,i+1) for i in range(49)]
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            legend()
        except: pass
    n=n+1
   
show()
ax=[subplot(7,8,i+1) for i in range(49)]
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: pass
    n=n+1
   
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: print('err')
    n=n+1
   
n=0
ax=[subplot(7,8,i+1) for i in range(49)]
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: print('err')
    n=n+1
   
ax=[subplot(4,14,i+1) for i in range(49)]
n=0
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: print('err')
    n=n+1
   
ax=[subplot(4,14,i+1) for i in range(49)]
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: print('err')
    n=n+1
   
n=0
ax=[subplot(4,14,i+1) for i in range(49)]
ax=[subplot(4,14,i+1) for i in range(49)]
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: print('err')
    n=n+1
   
ax=[subplot(14,4,i+1) for i in range(49)]
n=0
for ewl in ex_range:
    sp=[]
    for i in range(4):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: print('err')
    n=n+1
   
savefig('ZnO_spectra.pdf',format='pdf')
get_ipython().run_line_magic('logstart', '')
