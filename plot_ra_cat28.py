#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:10:02 2021

script to read in RA survey+mon catalogue for rags28 and plot useful info

@author: mikhail
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import scipy.spatial
#from tabulate import tabulate

def read_cat(FILE, add_paths=False):
    '''read catalogue into a dataframe.
    FORMAT:
  source       j2000     exper        date      time band polar      sta1      sta2 elev1 elev2     snr base_ml   base_cm     pa    ampl ampl_err  coeff  solint  data_len
0003-066  J0006-0623  raes03fb  2012-11-01  20:00:00    C    RR  EVPTRIYA  EFLSBERG  35.3  32.9  1743.8    32.5 2.023e+08  -79.3  0.0211   0.0012 1.0000      60       596
        '''
    df = pd.read_csv(FILE, comment='#', sep='\s+', engine='python')
    df = df.drop(['j2000', 'coeff', 'solint' , 'data_len'], axis=1)
    # combine date and time into a datetime-dtype column
    df.loc[:, 'start'] = df.date + ' ' + df.time
    df.start = pd.to_datetime(df.start)
    
    if add_paths:
        # add paths to logfiles etc.
        df.loc[:, 'path'] = df.loc[:, 'date'].str[0:7].str.replace('-', '_') + '/' +  df.loc[:, 'date'].str.replace('-', '_') + '_' + df.loc[:, 'exper']
        df.loc[:, 'antab'] = df.loc[:, 'date'].str[0:7].str.replace('-', '_') + '/' +  df.loc[:, 'date'].str.replace('-', '_') + '_' + df.loc[:, 'exper'] + '/' + df.loc[:, 'exper'] + df.loc[:, 'band'].str.lower() + '.antab'
    
    return df



def read_ulim(FILE):
    """Read upper limits from file. The format is different from that of the catalogue.
    b1950      j2000      exper_name band polar sta     start_time           solint base_ed base_ml  pa  elev  snr   ampl     snr_det  pfd   sefd     upper_lim
    0657+172 J0700+1709   rags28au    C    LL IRBENE16 2018-02-18T15:20:00   1200     3.5   717.3  -33.5 27.8 5.59 3.36e-05     6.85 5.6e-01 2875.0     0.118

    
    :FILE: a filename of the file with the upper limits
    :return: a Pandas DataFrame
    """
    
    df = pd.read_csv(FILE, comment='#', sep = '\s+', engine = 'python')
    
    # add sta2 column with RADIO-AS because all these upper limits are for baselines with RADIO-AS, as was requested
    df.loc[:, 'sta2'] = 'RADIO-AS'
    
    # make column names match those of the catalogue
    # sta2 = 'RADIO-AS'
    df.rename(columns = {'b1950': 'source', 'exper_name':'exper', 'sta': 'sta1', 'start_time':'start', 'ampl':'raw_ampl', 'upper_lim':'ampl'}, inplace = True) 
    df.start = pd.to_datetime(df.start)

    return df


  
def clean(df):
    '''remove known problematic data from the catalogue'''
    # no LCP data for IR in rags28a[ijklmnop]
    df = df.drop(df.loc[(((df.sta1 == 'IRBENE32') & (df.polar.str.startswith('L'))) | ((df.sta2 == 'IRBENE32') & (df.polar.str.endswith('L')))  ) & (df.exper.isin(['rags28ai','rags28aj','rags28ak','rags28al','rags28am','rags28an','rags28ao','rags28ap',])) ].index)
    # no LCP data for IR in rags28a[d]   - C-band
    df = df.drop(df.loc[(((df.sta1 == 'IRBENE32') & (df.polar.str.startswith('L')) & (df.band.str.startswith('C'))) | ((df.sta2 == 'IRBENE32') & (df.polar.str.endswith('L')))  & (df.band.str.startswith('C')) ) & (df.exper.isin(['rags28ad', 'rags28ah', 'rags28ag', 'rags28af', 'rags28ae'])) ].index)
    # too low ampl for no reason
    df = df.drop(df.loc[(((df.sta1 == 'TORUN') &  (df.band.str.startswith('C'))) | ((df.sta2 == 'TORUN')  & (df.band.str.startswith('C')) )) & (df.exper.isin(['rags28al'])) ].index)
    # AR-RA bad ampl in LL (it was a dual-pol 18 cm observation) rags28ac
    df = df.drop(df.loc[(((df.sta1 == 'ARECIBO') &  (df.band.str.startswith('L')) & (df.polar.str.endswith('L'))) | ((df.sta2 == 'ARECIBO')  & (df.band.str.startswith('L')) & (df.polar.str.endswith('L')) )) & (df.exper.isin(['rags28ac'])) ].index)
    
#    # TEST. Raise ampl on baselienes with TR by a factor of 2 in rags28a[ijk]
#    df.loc[(((df.sta1 == 'TORUN') &  (df.band.str.startswith('C'))) | ((df.sta2 == 'TORUN')  & (df.band.str.startswith('C')) )) & (df.exper.isin(['rags28ai','rags28aj','rags28ak'])),  'ampl'] = df.loc[(((df.sta1 == 'TORUN') &  (df.band.str.startswith('C'))) | ((df.sta2 == 'TORUN')  & (df.band.str.startswith('C')) )) & (df.exper.isin(['rags28ai','rags28aj','rags28ak'])) , 'ampl'] * 1

    
    
    # no LCP data from WB
    df = df.drop(df.loc[((df.sta1 == 'WSTRB-07') & (df.polar.str.startswith('L'))) | ((df.sta2 == 'WSTRB-07') & (df.polar.str.endswith('L')))  ].index )
    return df


def gauss(x, *p):
    ''' A Gaussian function to fit to the radplot data'''
    A, sigma = p
    return A*np.exp(-(x)**2/(2.*sigma**2))    # mu is set to 0 explicitly
    
def fit_gauss(df, zero_flux = None):
    '''Fit radplot data with gauss().
    Implicitly assume that in the input df there are columns 'base_ml' and 'ampl' 
    
    Args:
        df: DataFrame with columns base_ml and ampl
        zero_flux (float, default = None): Flux density at zero baselines take from 
            single-dish observations. Used only if set to non None. 
        
    Returns:
        xi, yi: coordinates of the fitted funstion
        
    '''
    # df here is local var to the function
    # sort df as base_ml
    df = df.sort_values(by = 'base_ml')
    
    if zero_flux is not None:
        print('Fitting with a zero flux set to {} Jy'.format(zero_flux))
        # make data symmetric around zero for a Gaussian fit adding zero for zero flux
        xdata =  np.append( -np.flip(df.base_ml) , 0) 
        xdata =  np.append( xdata , df.base_ml)
        ydata =  np.append(np.flip(df.ampl), zero_flux)
        ydata =  np.append(ydata,df.ampl)
    else:
        xdata =  np.append( -np.flip(df.base_ml) , df.base_ml) # make data symmetric around zero for a Gaussian fit
        ydata =  np.append(np.flip(df.ampl),df.ampl)
        
    p0 = [np.max(ydata), np.max(xdata)/3]  # for a Gaussian fit : p0 = [ Amplitude, sigma]. Position of the center is always zero in radplots.
    coeff, var_matrix = curve_fit(gauss, xdata, ydata, p0=p0) 
    xi = np.linspace(0, np.max(xdata), 1000)
    yi = gauss(xi, *coeff)

    return xi,yi
    

def onclick(event):
    '''show baseline parameters when clicked on a point'''
    
    # calculate all distances
    dl = np.sqrt((dd.base_ml - event.xdata)**2 /( dd.base_ml.max() - dd.base_ml.min()  )**2 + (dd.ampl - event.ydata)**2 / (dd.ampl.max() - dd.ampl.min() )**2)
    ind = dl.idxmin()
    print('Point at {:5.1f} ml, {:5.3f} Jy observed in {}-pol with {:8s}-{:8s} at {} {} ( {} )'.format(  *dd.loc[ind,['base_ml', 'ampl', 'polar', 'sta1', 'sta2', 'date', 'time', 'exper']] ))
    
    
def onclick_uv(event):
    '''show baseline parameters when clicked on a point'''
    
    # calculate all distances
    dl = np.sqrt((dd.base_ml* np.sin(np.deg2rad(dd.pa)) - event.xdata)**2 /( np.max(dd.base_ml* np.sin(np.deg2rad(dd.pa))) - np.min(dd.base_ml* np.sin(np.deg2rad(dd.pa)))  )**2 + (dd.base_ml* np.cos(np.deg2rad(dd.pa)) - event.ydata)**2 / (np.max(dd.ampl* np.cos(np.deg2rad(dd.pa))) - np.min(dd.ampl* np.cos(np.deg2rad(dd.pa))) )**2)
    ind = dl.idxmin()   # if both RCP and LCP are there, there is no control of which is chosen. Usually RCP
    inds = dl[dl==dl[ind]].index # indexes of all points at the same distance, either 1 or 2. 
    
    for i in inds:
        print('Point at {:5.1f} ml, {:5.3f} Jy observed in {}-pol with {:8s}-{:8s} at {} {} ( {} )'.format(  *dd.loc[i,['base_ml', 'ampl', 'polar', 'sta1', 'sta2', 'date', 'time', 'exper']] ))
    
# main
plot_type = 'uv'    
plot_type = 'radplot'
#
#FILE  = '/home/mikhail/sci/scatter/RA_catalog_rags28_2021-02-24.txt'
#FILE  = '/homes/mlisakov/sci/scatter/RA_catalog_rags28_2021-02-24.txt'
#FILE  = '/homes/mlisakov/sci/scatter/RA_catalog_rags28+raks18el_2021-04-16.txt'
#UPPERLIM = '/homes/mlisakov/sci/scatter/RA_rags28_nondet_uplims_2021-05-12.txt'
FILE =  'RA_catalog_rags28_2021-06-10.txt'
#FILE  = 'RA_catalog_rags28+raks18el_2021-04-16.txt'
UPPERLIM = 'RA_rags28_nondet_uplims_2021-05-12.txt'



df = read_cat(FILE)
df = clean(df)


# 2209+236 and 0657+172 separately
s1 = df.loc[df.source == '2209+236']
s2 = df.loc[df.source == '0657+172']
# C and L-bands separately
s1c = s1.loc[s1.band == 'C']
s1l = s1.loc[s1.band == 'L']
s2c = s2.loc[s2.band == 'C']
s2l = s2.loc[s2.band == 'L']





if UPPERLIM:
    du = read_ulim(UPPERLIM)    # upper limits
    # 2209+236 and 0657+172 separately
    s1U = du.loc[du.source == '2209+236']
    s2U = du.loc[du.source == '0657+172']
    # C and L-bands separately
    s1cU = s1U.loc[s1U.band == 'C']
    s1lU = s1U.loc[s1U.band == 'L']
    s2cU = s2U.loc[s2U.band == 'C']
    s2lU = s2U.loc[s2U.band == 'L']






# print summary per obscode

dd = s2l.sort_values(by = 'date')
dd = dd.sort_values(by = 'start')

print('SOURCE = {}, BAND = {}'.format(dd.source.unique()[0], dd.band.unique()[0]))



for o in dd.exper.unique():
    print(o)
    dd = dd.sort_values(by = 'base_ml')
    for i in dd[dd.exper == o].index:
        print('Baseline = {:5.1f} ml, {:6.1f} deg, {:5.3f} Jy observed in {}-pol with {:8s}-{:8s} at {} {} ( {} )'.format(  *dd.loc[i,['base_ml', 'pa',  'ampl', 'polar', 'sta1', 'sta2', 'date', 'time', 'exper']] ))

    
    


    print('\n\n')


# sort values for further convenience
# FOR RADPLOTS
if plot_type == 'radplot':
    for dd in [s1c, s1l, s2c, s2l]:
        dd.sort_values(by = 'base_ml', inplace = True)


if plot_type == 'time':
    for dd in [s1c, s1l, s2c, s2l]:
        dd.sort_values(by = ['date', 'time'], inplace=True)


if plot_type == 'uv':
    # do it for 0657+172, L-band to check TR calibration
    fig,ax = plt.subplots(1,1,subplot_kw=dict(aspect=1), sharex=True, sharey=True, constrained_layout=True)
    
    dd= s1l
    
    
    
    
    ddr = dd.loc[dd.polar == 'RR']
    x = -ddr.base_ml * np.sin(np.deg2rad(ddr.pa))
    y = ddr.base_ml * np.cos(np.deg2rad(ddr.pa))
    ax.plot(x,y, 'o', markersize = 14 , fillstyle = 'right', markeredgewidth=0.0, label = 'RR')
    
    maxx = np.max(np.abs(x))
    maxy = np.max(np.abs(y))
    maxmax = np.max([maxx, maxy])

    
    ddl = dd.loc[dd.polar == 'LL']
    x = -ddl.base_ml * np.sin(np.deg2rad(ddl.pa))
    y = ddl.base_ml * np.cos(np.deg2rad(ddl.pa))
    ax.plot(x,y, 'o', markersize = 14 , fillstyle = 'left', markeredgewidth=0.0, label = 'LL')
    
    maxx = np.max(np.abs(x))
    maxy = np.max(np.abs(y))
    maxmax = np.max([maxx, maxy, maxmax])
    ax.set_xlim([-maxmax, maxmax])
    ax.set_ylim([-maxmax, maxmax])
    
    ddTR = dd.loc[(dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN'), :]
    xTR = -ddTR.base_ml * np.sin(np.deg2rad(ddTR.pa))
    yTR = ddTR.base_ml * np.cos(np.deg2rad(ddTR.pa))
    
    
    ax.plot(xTR, yTR, '*r')
    
    
    
    ax.axvline(x=0, color = 'k', ls = '--')
    ax.axhline(y=0, color = 'k', ls = '--')
    
    ax.legend()
    cid = fig.canvas.mpl_connect('button_press_event', onclick_uv)





if plot_type == 'radplot':
    dd = s1l
    ddU = s1lU
    
    # Due to changed Tcal of TORUN.
    # Correct TR amplitudes (multiply by a factor sqrt(new Tcal / old Tcal)):
    # Old Tcals: RCP = 9.3, LCP = 10.2 K
    # in 2017:  RCP = 0.82   # seems to work well for 2209+236
    #           LCP = 0.84   # seems to work well for 2209+236
    # in 2018:  RCP = 2.22
    #           LCP = 1.98
    
    dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date < '2017-12-31') & (dd.polar == 'RR')  , 'ampl'] =\
        dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date < '2017-12-31') & (dd.polar == 'RR')  , 'ampl'] * 0.82
    dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date < '2017-12-31') & (dd.polar == 'LL')  , 'ampl'] =\
        dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date < '2017-12-31') & (dd.polar == 'LL')  , 'ampl'] * 0.84
    dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date > '2017-12-31') & (dd.polar == 'RR')  , 'ampl'] =\
        dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date > '2017-12-31') & (dd.polar == 'RR')  , 'ampl'] * 2.22
    dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date > '2017-12-31') & (dd.polar == 'LL')  , 'ampl'] =\
        dd.loc[((dd.sta1 == 'TORUN') | (dd.sta2 == 'TORUN')) & (dd.date > '2017-12-31') & (dd.polar == 'LL')  , 'ampl'] * 1.98
#     
    
#    dd= dd.loc[dd.date < '2018-03-01']
#    dd= dd.loc[dd.date > '2018-03-01']
    
#    dd = dd.loc[(dd.exper == 'rags28ax') & (dd.polar == 'RR')]
#    dd = dd.loc[(dd.sta1 != 'TORUN') & (dd.sta2 != 'TORUN') & (dd.sta1 != 'WSTRB-07') & (dd.sta2 != 'WSTRB-07')]
#    dd = dd.loc[ (dd.sta1 != 'WSTRB-07') & (dd.sta2 != 'WSTRB-07')]

    band = dd.band.unique()[0].lower()
    source = dd.source.unique()[0]
    
    if source == '0657+172':
        if band == 'l':
            zero_flux = 0.82
        if band == 'c':
            zero_flux = 0.67
    elif source == '2209+236':
        if band == 'l':
            zero_flux = 0.82
        if band == 'c':
            zero_flux = 0.83
    else:
        zero_flux = None

    
    
    
    dfit = dd.loc[ (dd.sta1 != 'WSTRB-07') & (dd.sta2 != 'WSTRB-07')]
#    xi, yi = fit_gauss(dd, zero_flux = zero_flux)
    xi, yi = fit_gauss(dfit, zero_flux = zero_flux)
    
    
    # plot one source at a time
    fig,ax = plt.subplots(1,1, figsize = (12,8))
    x = dd.base_ml
    y = dd.ampl
    
    
    if zero_flux is not None:
        ax.plot(0, zero_flux, '*r', label = 'Zero flux')
        
    ax.plot(x.loc[dd.polar == 'RR'],y.loc[dd.polar == 'RR'], 'o', label = '{}, {}-band, RR'.format(source, band.upper()))
    ax.plot(x.loc[dd.polar == 'LL'],y.loc[dd.polar == 'LL'], 'o', label = '{}, {}-band, LL'.format(source, band.upper()))
    ax.plot(xi, yi)
    
    if UPPERLIM: 
        
        x = ddU.loc[ddU.sta1.isin(['ARECIBO', 'EFLSBERG', 'GBT-VLBA']) , 'base_ml']
#        x = ddU.base_ml
        y = ddU.loc[ddU.sta1.isin(['ARECIBO', 'EFLSBERG', 'GBT-VLBA']) , 'ampl']
        
        ax.plot(x.loc[ddU.polar == 'RR'],y.loc[ddU.polar == 'RR'], 'v', label = 'UPPER lim {}, {}-band, RR'.format(source, band.upper()))
        ax.plot(x.loc[ddU.polar == 'LL'],y.loc[ddU.polar == 'LL'], 'v', label = 'UPPER lim {}, {}-band, LL'.format(source, band.upper()))
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax.legend()



