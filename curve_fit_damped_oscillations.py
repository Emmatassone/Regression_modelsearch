import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Parameters
column1,column2,column3='e','final-mass','initial_m1'

path='NS_Simulations_554.dat'
data = pd.read_csv(path)
data = data.astype(float)

def poly2_damped_oscillator(t,amplitude,f1,f2,phase,decay,k):#,amplitude2,frequency2,phase2,decay2
    return amplitude*np.exp(decay*t)*np.cos(f1*t+f2*t**2 + phase)+k

def piecewise_function1(e,x0,A,k1):
    m1o50_s11_params={'amplitude': 0.00014887556639812068, 'f1': -154.7170957190035, 'f2': 174.87960460832457, 'phase': 74.97462188181812, 'decay': 11.520402722919949, 'k': 0.9515427928245563}
    if e<=x0:
        return poly2_damped_oscillator(e,**m1o50_s11_params)
    else:
        w1=m1o50_s11_params['f1']
        w2=m1o50_s11_params['f2']
        return k1+A*np.arctan(w1*e+w2*e**2)

def piecewise_function2(e,x0,A,w1,phi,k1):
    m1o50_s11_params={'amplitude': 0.00014887556639812068, 'f1': -154.7170957190035, 'f2': 174.87960460832457, 'phase': 74.97462188181812, 'decay': 11.520402722919949, 'k': 0.9515427928245563}
    #cosh best fit parameters np.array([-2.8246e-04 ,6.64790028,-5.83840220,0.97877643])
    if e<=x0:
        return poly2_damped_oscillator(e,**m1o50_s11_params)
    else:
        return k1+A*np.cosh(w1*e**2+phi)

def piecewise_function3(e,x0,a0,a1,a2,a3,a4):#,
    [amplitude,f1,f2,phase,decay,k]=[2.6161e-05,926.153622,-741.2561,-239.0159,9.2364,0.95111]
    
    if e<=x0:
        return amplitude*np.exp(decay*e)*np.cos(f1*e+f2*e**2 + phase)+k
    else:
        return np.poly1d([a0,a1,a2,a3,a4])(e)#
    
piecewise_function_vec=np.vectorize(piecewise_function2)
table['e']=round(table['e'], 4)
table=table.sort_values(column1)
separation_classes=np.unique(round(table['initial-separation'],0).values)
sep0=separation_classes[3]
fixed_sep=table[round(table['initial-separation'],0)==sep0]
column3_classes=np.unique(fixed_sep[column3].values)
plt.rcParams["figure.figsize"] = (15,5)

parameter=column3_classes[-1]
fixed_parameter=fixed_sep[fixed_sep[column3]==parameter]
fixed_parameter=fixed_parameter[fixed_parameter[column2]>0.92]#filter one point with non usual final-mass 
x_to_fit=fixed_parameter[column1].values
y_to_fit=fixed_parameter[column2].values

from scipy.optimize import curve_fit

guess = np.array([0.25,-2.8246e-04 ,6.64790028,-5.83840220,0.97877643])
#guess = np.array([0.56,-1.192,4.5096,-6.48197,4.209577,-0.0556])
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
plt.savefig('piecewisefunction_coshModel_'+column1+'_vs_'+column2+'_sep_'+str(sep0)+'_initialm1_'+str(parameter)+'.png')