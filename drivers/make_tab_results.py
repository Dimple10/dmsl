'''
makes latex table with results
'''

## load packages
import numpy as np
import pickle
import astropy.units as u
import dill
import pandas as pd

from dmsl.convenience import *
from dmsl.paths import *
from dmsl.survey import Roman

## set lens types and surveys to loop through
lenstypes = ['ps', 'gaussian', 'nfw']
surveys = ['Roman']
labels = {'ps':[r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$', r'$\rm{Fraction~of~DM}$'],
        'gaussian': [r'$\log_{10} M_l~[\mathrm{M}_{\odot}]$',
            r'$\log_{10} R_0~[\rm{pc}]$',
            r'$\rm{Fraction~of~DM}$']}

def get_chains(lenstypes, surveys):
    chains = {}
    for survey in surveys:
        for l in lenstypes:
            tstring = f'{survey}_{l}'
            f = f'{FINALDIR}pruned_samples_{survey}_{l}_3_4_2.pkl'
            with open(f,'rb') as buff:
                chains[tstring] = dill.load(buff)
            tstring = f'{survey}_{l}_frac'
            f = f'{FINALDIR}pruned_samples_{survey}_{l}_frac_3_4_2.pkl'
            with open(f,'rb') as buff:
                chains[tstring] = dill.load(buff)
    return chains

def get_nums_for_table(chain, percent=95, twosided=False, side=None):
    ave = np.average(chain)
    if twosided:
        confup = np.percentile(chain, 100-(100-percent)/2.)
        confdown = np.percentile(chain, (100-percent)/2.)
        conf = [ave, confup, confdown]
        sgn = -1
    if twosided == False and side == 'up':
        conf = np.percentile(chain, percent)
        sgn = 1
    if twosided == False and side == 'down':
        conf = np.percentile(chain, 100-percent)
        sgn = 0
    return conf, sgn

def write_table(data, firstcol, cols):
    df = pd.DataFrame(data=data, columns=cols[1:])
    df.insert(loc=0, column=cols[0], value=firstcol)
    df.to_latex(PAPERDIR+'tab_results.tex', index=False,
               column_format='l|cccc', label='tab:results', caption='Forecasted 95\% Limits on DM Substructure Using the Roman Space Telescope EML Survey',
               escape=False,
               bold_rows=True)
    return df

chains = get_chains(lenstypes, surveys)
colnames = ['Mass Profile Type',
        'Mass Limits [$\\log_{{10}} \\rm{M}_{\\odot}$]',
        'Radius Limits [$\\log_{{10}} \\rm{pc}$]',
        'Concentration Limits [$\\log_{{10}} c_{200}$]',
        'Fraction Limits']
nicelenstypes = ['Point Source', 'Point Source + Fraction',
        'Gaussian','Gaussian + Fraction', 'NFW', 'NFW + Fraction']

d = np.empty((6,4), dtype='U32')
counter = 0
for l in lenstypes:
    kstrings = [f'{l}', f'{l}_frac']
    for kstring in kstrings:
        nums = np.empty((2, 4))
        nums[:] = np.nan
        for i in range(0,np.shape(chains['Roman_'+kstring])[1]):
            coli = i
            side = 'up'
            twosided = False
            if ('frac' in kstring) and (i == np.shape(chains['Roman_'+kstring])[1] - 1):
                coli = -1
            if (l=='gaussian') and(i==1):
                side = 'down'
            if (l=='nfw') and (i==1):
                coli = 2
            nums[:,coli] = get_nums_for_table(chains['Roman_'+kstring][:,i], side=side, twosided = twosided)
        info = []
        for i in range(0, np.shape(nums)[1]):
            if np.all(np.isnan(nums[:,i])):
                info.append('--')
            elif (l=='ps') and (i==0):
                upconf, _ = get_nums_for_table(chains['Roman_'+kstring][:,i], side='up')
                downconf, _ = get_nums_for_table(chains['Roman_'+kstring][:,i], side='down')
                info.append(f'$>{downconf:.1f}~\& <{upconf:.1f}$')
            else:
                if nums[1,i] == 1:
                    sgn = '<'
                if nums[1,i] == 0:
                    sgn = '>'
                info.append(f'${sgn}{nums[0,i]:.2f}$')
        d[counter] = info
        counter +=1
write_table(d, nicelenstypes, colnames)
with open(PAPERDIR+'tab_results.tex', "r") as f:
  filedata = f.read()

# Replace the table with table*
text = filedata.replace("\\begin{table}", "\\begin{table*}")
text = text.replace("\\end{table}", "\\end{table*}")

# Write the file out again
with open(PAPERDIR+'tab_results.tex', "w") as f:
  f.write(text)
with open(FINALDIR+'tab_results.tex', "w") as f:
    f.write(text)
