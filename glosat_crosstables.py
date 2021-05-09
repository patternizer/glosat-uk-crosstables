#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: glosat_crosstables.py
#------------------------------------------------------------------------------
# Version 0.1
# 7 May, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Numerics and dataframe libraries:
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import calendar as cal
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.path as mpath
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cmocean
# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.feature import NaturalEarthFeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# Math libraries:
import scipy
from sklearn import datasets, linear_model

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

use_darktheme = True
plot_crosstable = True
nz = 4
topn = 3
fontsize = 14
region = 'scotland'
#region = 'uk'
#region = 'east-anglia'
month = 4
months = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
x_variable = 'tmean'
#x_variable = 'tmin'
#x_variable = 'tmax'
#x_variable = 'rainfall'
#x_variable = 'raindays'
#x_variable = 'frostdays'
#x_variable = 'sunshine'

#y_variable = 'tmean'
#y_variable = 'tmin'
#y_variable = 'tmax'
y_variable = 'rainfall'
#y_variable = 'raindays'
#y_variable = 'frostdays'
#y_variable = 'sunshine'

f_x = 'DATA/' + region + '-year-' + x_variable + '.txt'
f_y = 'DATA/' + region + '-year-' + y_variable + '.txt'
if (x_variable == 'tmean'): x_str_lo = 'COLD'; x_str_hi = 'WARM'
if (x_variable == 'tmin'): x_str_lo = 'LOW'; x_str_hi = 'HIGH'
if (x_variable == 'tmax'): x_str_lo = 'LOW'; x_str_hi = 'HIGH'
if (x_variable == 'rainfall'): x_str_lo = 'DRY'; x_str_hi = 'WET'
if (x_variable == 'raindays'): x_str_lo = 'CLEAR'; x_str_hi = 'RAINY'
if (x_variable == 'frostdays'): x_str_lo = 'CLEAR'; x_str_hi = 'FROSTY'
if (x_variable == 'sunshine'): x_str_lo = 'DIM'; x_str_hi = 'BRIGHT'    
if (y_variable == 'tmean'): y_str_lo = 'COLD'; y_str_hi = 'WARM'
if (y_variable == 'tmin'): y_str_lo = 'LOW'; y_str_hi = 'HIGH'
if (y_variable == 'tmax'): y_str_lo = 'LOW'; y_str_hi = 'HIGH'
if (y_variable == 'rainfall'): y_str_lo = 'DRY'; y_str_hi = 'WET'
if (y_variable == 'raindays'): y_str_lo = 'FEW'; y_str_hi = 'MANY'
if (y_variable == 'frostdays'): y_str_lo = 'CLEAR'; y_str_hi = 'FROSTY'
if (y_variable == 'sunshine'): y_str_lo = 'DIM'; y_str_hi = 'BRIGHT'    

# Calculate current time

now = datetime.now()
currentmn = str(now.month)
if now.day == 1:
    currentdy = str(cal.monthrange(now.year,now.month-1)[1])
    currentmn = str(now.month-1)
else:
    currentdy = str(now.day-1)
if int(currentdy) < 10:
    currentdy = '0' + currentdy    
currentyr = str(now.year)
if int(currentmn) < 10:
    currentmn = '0' + currentmn
currenttime = str(currentdy) + '_' + currentmn + '_' + currentyr
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr

#------------------------------------------------------------------------------
# THEME
#------------------------------------------------------------------------------

if use_darktheme == True:
    matplotlib.rcParams['text.usetex'] = True
#   matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ] 
    plt.rc('text',color='white')
    plt.rc('lines',color='white')
    plt.rc('patch',edgecolor='white')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('axes',edgecolor='lightgray')
    plt.rc('axes',facecolor='black')
    plt.rc('axes',labelcolor='white')
    plt.rc('figure',facecolor='black')
    plt.rc('figure',edgecolor='black')
    plt.rc('savefig',edgecolor='black')
    plt.rc('savefig',facecolor='black')
else:
    matplotlib.rcParams['text.usetex'] = True
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]
    plt.rc('text',color='black')
    plt.rc('lines',color='black')
    plt.rc('patch',edgecolor='black')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='black')
    plt.rc('ytick',color='black')
    plt.rc('axes',edgecolor='black')
    plt.rc('axes',facecolor='white')
    plt.rc('axes',labelcolor='black')
    plt.rc('figure',facecolor='white')
    plt.rc('figure',edgecolor='black')
    plt.rc('savefig',edgecolor='black')
    plt.rc('savefig',facecolor='white')

#----------------------------------------------------------------------------
# CREDITS
#----------------------------------------------------------------------------

sourcestr = r'$\textbf{Data}$' + ': https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-and-regional-series/'        
authorstr = r'$\textbf{Graphic}$' + ': Michael Taylor, CRU/UEA' + ' -- ' + titletime

#----------------------------------------------------------------------------
# LOAD: dependent variable X
#----------------------------------------------------------------------------

nheader = 6
f = open(f_x)
lines = f.readlines()
years = []
vals = []
for i in range(nheader,len(lines)):
    words = lines[i].split()    
    year = int(words[0])
    if (words[month] == '---'):
    	val = mp.nan
    else:
    	val = float(words[month])
    years.append(year)
    vals.append(val)
f.close()    
years = np.array(years)
vals = np.array(vals)
mu = np.mean(vals)
sigma = np.std(vals)
x_vals = [ ((vals[i]-mu)/sigma) for i in range(len(vals)) ]
dx = pd.DataFrame({'year':years, x_variable:x_vals})

#----------------------------------------------------------------------------
# LOAD: independent variable Y
#----------------------------------------------------------------------------

nheader = 6
f = open(f_y)
lines = f.readlines()
years = []
vals = []
for i in range(nheader,len(lines)):
    words = lines[i].split()    
    year = int(words[0])
    val = float(words[month])
    years.append(year)
    vals.append(val)
f.close()    
years = np.array(years)
vals = np.array(vals)
mu = np.mean(vals) 
sigma = np.std(vals)
y_vals = [ ((vals[i]-mu)/sigma) for i in range(len(vals)) ]
dy = pd.DataFrame({'year':years, y_variable:y_vals})

if (x_variable == y_variable):
    ds = dx.copy()
    ds[y_variable] = y_vals
else:
    ds = dx.merge(dy, on='year')

#----------------------------------------------------------------------------
# CALCULATE: Top 5 in each quadrant
#----------------------------------------------------------------------------

Q1 = ds[(ds[x_variable]<0) & (ds[y_variable]>0)]
Q2 = ds[(ds[x_variable]>0) & (ds[y_variable]>0)]
Q3 = ds[(ds[x_variable]>0) & (ds[y_variable]<0)]
Q4 = ds[(ds[x_variable]<0) & (ds[y_variable]<0)]
N_Q1 = np.linalg.norm(Q1[[x_variable,y_variable]].values,axis=1)
N_Q2 = np.linalg.norm(Q2[[x_variable,y_variable]].values,axis=1)
N_Q3 = np.linalg.norm(Q3[[x_variable,y_variable]].values,axis=1)
N_Q4 = np.linalg.norm(Q4[[x_variable,y_variable]].values,axis=1)
Q1_topn = np.argsort(N_Q1)[-topn:][::-1]
Q2_topn = np.argsort(N_Q2)[-topn:][::-1]
Q3_topn = np.argsort(N_Q3)[-topn:][::-1]
Q4_topn = np.argsort(N_Q4)[-topn:][::-1]
if (x_variable == y_variable):
    Q1_topn = Q2_topn; Q1 = Q2
    Q3_topn = Q4_topn; Q3 = Q4
Q1_rank = [ Q1.iloc[Q1_topn[i]].year.astype(int) for i in range(topn) ]
Q2_rank = [ Q2.iloc[Q2_topn[i]].year.astype(int) for i in range(topn) ]
Q3_rank = [ Q3.iloc[Q3_topn[i]].year.astype(int) for i in range(topn) ]
Q4_rank = [ Q4.iloc[Q4_topn[i]].year.astype(int) for i in range(topn) ]

#----------------------------------------------------------------------------
# PLOT: timeseries
#----------------------------------------------------------------------------

if plot_crosstable == True:

    print('plot_crosstable ...')

    figstr = region + '-' + str(month).zfill(2) + '-' + x_variable + '-' + y_variable + '.png'
    titlestr = region.upper() + ' (' + months[month] + ') ' + x_variable.capitalize() + ' versus ' + y_variable.capitalize() + ' anomalies'
        
    fig, ax = plt.subplots(figsize=(15,10))              
    plt.fill([0,nz,nz,0], [0,0,nz,nz], 'red', alpha=0.7, edgecolor=None, zorder=0)    
    plt.fill([0,-nz,-nz,0], [0,0,nz,nz], 'linen', alpha=0.5, edgecolor=None, zorder=0)    
    plt.fill([0,nz,nz,0], [0,0,-nz,-nz], 'lime', alpha=0.5, edgecolor=None, zorder=0)    
    plt.fill([0,-nz,-nz,0], [0,0,-nz,-nz], 'blue', alpha=0.7, edgecolor=None, zorder=0)    
    plt.scatter(ds[x_variable], ds[y_variable], s=50, c=ds.year, cmap='Greys', zorder=10)
    idx2021 = ds[ds['year']==2021].index[0]
    plt.plot(ds[x_variable][idx2021], ds[y_variable][idx2021], 'o', color='cyan',  linewidth=3, markersize=30, markerfacecolor='none', markeredgewidth=3, zorder=9)
    cb = plt.colorbar(orientation="vertical", shrink=0.8, pad=0.05, extend='both')
    
    for i in range(topn):       
    
        x = ds[ds.year==Q1_rank[i]][x_variable]
        y = ds[ds.year==Q1_rank[i]][y_variable]            
        plt.text(x,y,str(Q1_rank[i]),color='black', fontsize=fontsize, zorder=20)
        plt.plot(x,y, 'o', color='orange', linewidth=2, markersize=15, markerfacecolor='none', markeredgewidth=2, zorder=9)
        x = ds[ds.year==Q2_rank[i]][x_variable]
        y = ds[ds.year==Q2_rank[i]][y_variable]            
        plt.text(x,y,str(Q2_rank[i]),color='black', fontsize=fontsize, zorder=20)
        plt.plot(x,y, 'o', color='orange', linewidth=2, markersize=15, markerfacecolor='none', markeredgewidth=2, zorder=9)
        x = ds[ds.year==Q3_rank[i]][x_variable]
        y = ds[ds.year==Q3_rank[i]][y_variable]            
        plt.text(x,y,str(Q3_rank[i]),color='black', fontsize=fontsize, zorder=20)
        plt.plot(x,y, 'o', color='orange', linewidth=2, markersize=15, markerfacecolor='none', markeredgewidth=2, zorder=9)
        x = ds[ds.year==Q4_rank[i]][x_variable]
        y = ds[ds.year==Q4_rank[i]][y_variable]            
        plt.text(x,y,str(Q4_rank[i]),color='black', fontsize=fontsize, zorder=20)
        plt.plot(x,y, 'o', color='orange', linewidth=2, markersize=15, markerfacecolor='none', markeredgewidth=2, zorder=9)
    
    cb.ax.set_title(r'year', fontsize=fontsize)        
    cb.ax.tick_params(labelsize=fontsize)            
    ax.axhline(0, color='black', lw=2, zorder=5)
    ax.axvline(0, color='black', lw=2, zorder=5)
    lo_str_lo = x_str_lo + r'$\, \& \,$' + y_str_lo
    lo_str_hi = x_str_lo + r'$\, \& \,$' + y_str_hi
    hi_str_hi = x_str_hi + r'$\, \& \,$' + y_str_hi
    hi_str_lo = x_str_hi + r'$\, \& \,$' + y_str_lo
    if (x_variable == y_variable):
        if x_variable == 'rainfall':
            plt.text((-nz+0.5), (nz-0.5), x_str_hi, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='left')
            plt.text((nz-0.5), (-nz+0.5), x_str_lo, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='right')    
        else:
            plt.text((nz-0.5), (nz-0.5), x_str_hi, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='right')
            plt.text((-nz+0.5), (-nz+0.5), x_str_lo, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='left')            
    else:
        plt.text((-nz+0.5), (nz-0.5), lo_str_hi, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='left')
        plt.text((nz-0.5), (nz-0.5), hi_str_hi, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='right')
        plt.text((-nz+0.5), (-nz+0.5), lo_str_lo, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='left')
        plt.text((nz-0.5), (-nz+0.5), hi_str_lo, color='black', fontweight="bold", fontsize=24, bbox = dict(boxstyle = "square", fc = "lightgrey", alpha=0.5), ha='right')
    ax.set_xlim(-nz,nz)
    ax.set_ylim(-nz,nz)
    ax.xaxis.grid(True, which='major', alpha=0.5, zorder=1)        
    ax.yaxis.grid(True, which='major', alpha=0.5, zorder=1)      
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    plt.setp( ax.get_xticklabels(), visible=False)
    plt.setp( ax.get_yticklabels(), visible=False)
    if use_darktheme == True:            
        fig.suptitle(titlestr, fontsize=30, color='white', fontweight='bold')            
        plt.annotate(sourcestr, xy=(200,45), xycoords='figure pixels', color='white', fontsize=fontsize)             
        plt.annotate(authorstr, xy=(200,20), xycoords='figure pixels', color='white', fontsize=fontsize, bbox=dict(boxstyle="square, pad=0.3", fc='black',  edgecolor='white', linewidth=0.2))             
    else:            
        fig.suptitle(titlestr, fontsize=30, color='black', fontweight='bold')            
        plt.annotate(sourcestr, xy=(200,45), xycoords='figure pixels', color='black', fontsize=fontsize)                 
        plt.annotate(authorstr, xy=(200,20), xycoords='figure pixels', color='black', fontsize=fontsize, bbox=dict(boxstyle="square, pad=0.3", fc='white',  edgecolor='black', linewidth=0.2))             
    fig.subplots_adjust(left=0.2, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    plt.savefig(figstr)
    plt.close(fig)
    
# -----------------------------------------------------------------------------
print('** END')
