import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Parameters
column1,column2,column3='e','final-chi','initial_m1'

path='/home/emmanuel/Dropbox/Statistics/DataAnalysis/NS_Simulations_554.dat'
table=pd.read_csv(path).astype('float')


def piecewise_function1(e,x0,A,w1,k1):
    [amplitude,f1,f2,phase,decay,k]=[6.7907e-04,-150.841518,161.886944,91.9933495,10.0860683,0.68690750]
    if e<=x0:
        return amplitude*np.exp(decay*e)*np.cos(f1*e+f2*e**2 + phase)+k
    else:
        return k1+A*np.sqrt(1-w1*e**2)

def piecewise_function2(e,x0,a0,a1,a2):#,a3,a4
    [amplitude,f1,f2,phase,decay,k]=[1.4632e-04,811.404513 ,-616.16,-198.65,8.1574,0.68756017]
    if e<=x0:
        return amplitude*np.exp(decay*e)*np.cos(f1*e+f2*e**2 + phase)+k
    else:
        return np.poly1d([a0,a1,a2])(e)
    
def piecewise_function3(e,x0,A,w1,phi,k1):
    [amplitude,f1,f2,phase,decay,k]=[1.4632e-04,811.404513 ,-616.16,-198.65,8.1574,0.68756017]
    #cosh best fit parameters np.array([-2.8246e-04 ,6.64790028,-5.83840220,0.97877643])
    if e<=x0:
        return amplitude*np.exp(decay*e)*np.cos(f1*e+f2*e**2 + phase)+k
    else:
        return k1+A*np.cosh(w1*e**2+phi)
    
    
piecewise_function_vec=np.vectorize(piecewise_function2)
table['e']=round(table['e'], 4)
table=table.sort_values(column1)
separation_classes=np.unique(round(table['initial-separation'],0).values)
sep0=separation_classes[-1]
fixed_sep=table[round(table['initial-separation'],0)==sep0]
column3_classes=np.unique(fixed_sep[column3].values)
plt.rcParams["figure.figsize"] = (15,5)

parameter=column3_classes[-1]
fixed_parameter=fixed_sep[fixed_sep[column3]==parameter]
x_to_fit=fixed_parameter[column1].values
y_to_fit=fixed_parameter[column2].values

from scipy.optimize import curve_fit

#guess = np.array([0.56,-0.003,100,-100,0])
guess = np.array([0.55,0.68445763,0.2523555,0.68445763])
params, covs = curve_fit(piecewise_function_vec, x_to_fit, y_to_fit,guess)
print("params: ", params) 
print("covariance: ", covs) 

fig,axs=plt.subplots(1)
axs.scatter(x_to_fit,y_to_fit,label='original')

y_ev=[piecewise_function_vec(x,*params) for x in x_to_fit]
axs.plot(x_to_fit, y_ev, '--', label='fit',c='crimson')
axs.legend(loc='best')
axs.grid()

plt.title('initial sep '+ str(sep0))
plt.savefig('fit_piecewisefunction_polyModel_'+column1+'_vs_'+column2+'_sep_'+str(sep0)+'_initialm1_'+str(parameter)+'.png')