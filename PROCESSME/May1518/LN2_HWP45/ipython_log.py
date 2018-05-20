# IPython log file

s=pd.HDFStore('measLaserSpectra1526391046.h5')
import pandas as pd
s=pd.HDFStore('measLaserSpectra1526391046.h5')
s.keys()
ex_range=np.arange(950,1165,5)
len(ex_range)
ax=[subplot(12,4,i+1) for i in range(49)]
ax=[subplot(12,4,i+1) for i in range(43)]
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
    
ax=[subplot(12,4,i+1) for i in range(43)]
n=0
for ewl in ex_range:
    sp=[]
    for i in range(2):
        try:
            spectra = s['/spectra_'+str(ewl)+'_NP'+str(i)]
            ax[n].plot(spectra.index,spectra.intens,label='NP'+str(i))
            ax[n].set_title("@"+str(ewl)+'nm')
            
        except: print('err')
    n=n+1
    
savefig('LN_spectra.pdf',format='pdf')
savefig('LN_spectra.pdf',format='pdf')
savefig('LN_spectra.pdf',format='pdf')
get_ipython().run_line_magic('logstart', '')
