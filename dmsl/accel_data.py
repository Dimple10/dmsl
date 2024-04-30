'''
Makes fake data vector
'''

import os
import numpy as np
import pandas as pd

from dmsl.paths import *
from dmsl.convenience import *
from dmsl.plotting import *

from dmsl.prior import *
import dmsl.lensing_model as lm
import dmsl.mass_profile as mp
import dmsl.mass_function as mf
import dmsl.galaxy_subhalo_dist as gsh
import scipy.stats
import time
import scipy.interpolate
from scipy.interpolate import UnivariateSpline
import astropy.units as u

seed = 0

class AccelData():

    def __init__(self, survey,nstars=1000, ndims=1, wdm=False):
        rng = np.random.default_rng(seed=seed)
        if wdm:
            cdm = mf.CDM_Test()
            nfw = mp.NFW(Ml=1.e5*u.Msun, c200= 2.5)
            self.cdm_alphal = self.samplealphal([np.log10(13),np.log10(3.26 * 10 ** -5),-1.9,np.log10(2.57 * 10 ** 7)],survey,nstars, nfw, cdm)
            data = pd.DataFrame(self.cdm_alphal.value,columns=['a_x', 'a_y'])
            print('CDM data:',type(self.cdm_alphal), self.cdm_alphal)
                #sampler.make_new_mass([2.5,np.log10(3.26 * 10 ** -5),-1.9,np.log10(2.57 * 10 ** 7)], 'CDM','nfw')
        else:
            if ndims == 2:
                data = pd.DataFrame(rng.standard_normal(size=(nstars,ndims))*survey.alphasigma.value,
                        columns=['a_x', 'a_y'])
            else:
                data = pd.DataFrame(rng.standard_normal(size=(nstars,ndims))*survey.alphasigma.value,
                        columns=['a'])
        fileinds = [np.log10(nstars), ndims, survey.alphasigma.value]
        filepath = make_file_path(STARDATADIR, fileinds,
                extra_string=f'{survey.name}',ext='.dat')
        data.to_csv(filepath, index=False)
        print("Wrote to {}".format(filepath))
        self.plot_data(data,survey, nstars, ndims)
        self.data = data

    def plot_data(self, data, survey, nstars, ndims):
        fileinds = [np.log10(nstars), ndims, survey.alphasigma.value]
        filepath = make_file_path(STARDATADIR, fileinds,
                extra_string=f'{survey.name}_hist',ext='.png')
        print(filepath)
        print(data.shape)
        make_histogram(data.to_numpy(),int(len(data)/100) + 1, r'\alpha', filepath)

    def lnlike(self,pars,survey,nstars,mptype,mftype):
        # if ~np.isfinite(self.logprior(pars)):
        #     return -np.inf
        alphal = self.samplealphal(pars,survey,nstars,mptype,mftype)
        #FIXME
        # if self.massprofile.type == 'ps':
        #     # if ps, need to do snr check and re-sample if any have too high snr. this stops walkers from getting too stuck.
        #      alphal = self.snr_check(alphal, pars)
        #      if alphal == "error":
        #          return -np.inf
        if np.any(np.isnan(alphal)): #FIXME Should not be needed!
            return -np.inf
        try:
            #print('Trying')
            diff = alphal.value - self.data
        except:
            return -np.inf
        chisq = -0.5 * np.sum((diff)**2 / survey.alphasigma.value**2 -np.log(2 * np.pi * survey.alphasigma.value**2))
        return chisq

    def samplealphal(self, pars, survey, nstars, mptype=None,mftype=None):
        rv = gsh.initialize_dist(target=survey.target,
                                 rmax=survey.maxdlens.to(u.kpc).value)
        rdist = rv
        bs = np.logspace(-8, np.log10(np.sqrt(2)*survey.fov_rad),nstars)

        newmassprofile, newmassfunction = self.make_new_mass(pars, mptype,mftype)
        nlens = newmassfunction.n_l
        if sum(nlens) ==0:
            nlens[0] = 1
        priorpdf = pdf(bs, a1=survey.fov_rad, a2=survey.fov_rad,
            n=sum(nlens))
        #print(priorpdf, self.bs, sum(nlens))
        if np.any(np.isnan(priorpdf)):
            return -np.inf

        priorpdfspline = UnivariateSpline(np.log10(bs[priorpdf>0]),
                np.log10(priorpdf[priorpdf>0]), ext='zeros', s=0)
        dists = rdist.rvs(nstars) * u.kpc
        x = bs
        y = 10**priorpdfspline(np.log10(bs))
        sci_s = scipy.interpolate.interp1d(x, y, fill_value='extrapolate')
        sci = sci_s(x)
        beff = np.random.choice(x, nstars, p=sci / sum(sci)) * dists
        vl = scipy.stats.truncnorm.rvs(a=0, b=550., loc=0.,scale =220, size=nstars)
        bvec = np.zeros((nstars, 2))
        vvec = np.zeros((nstars, 2))
        btheta = np.random.rand(nstars)* 2. * np.pi
        vtheta = np.random.rand(nstars)* 2. * np.pi
        bvec[:, 0] = beff * np.cos(btheta)
        bvec[:, 1] = beff * np.sin(btheta)
        vvec[:, 0] = vl * np.cos(vtheta)
        vvec[:, 1] = vl * np.sin(vtheta)

        bvec *= u.kpc
        vvec *= u.km / u.s
        ## get alphal given sampled other params.
        start = time.perf_counter()
        alphal = lm.alphal(newmassprofile, bvec, vvec)
        end = time.perf_counter()
        #print(f'Time taken for lensing model: {(end-start):.6f} second')
        return alphal

    def make_new_mass(self,pars,mpt,mft): #FIXME For mf does this need to be mf + mp? or just mf?
        mptype = mpt.type
        mftype = mft.Name
        kwargs = mpt.kwargs
        i = 1
        if mptype == 'ps':
            i = 0
        if mftype == 'PowerLaw':
            logalpha = pars[i+0]
            logM0 = pars[i+1]
            newmf = mf.PowerLaw(logM_0=logM0, logalpha=logalpha)
        elif mftype == 'Tinker':
            A = pars[i+0]
            a = pars[i+1]
            b = pars[i+2]
            c = pars[i+3]
            k_b = pars[i+4]
            n_b = pars[i+5]
            k_s = pars[i+6]
            newmf = mf.Tinker(A= A, a= a, b= b, c= c, k_b=k_b, n_b=n_b, k_s=k_s)
        elif mftype == 'CDM':
            loga = pars[i+0]
            b = pars[i+1]
            logc = pars[i+2]
            newmf = mf.CDM_Test(loga = loga, b = b,logc = logc)
        elif mftype == 'WDM Stream':
            m_wdm = pars[i+0]
            gamma = pars[i+1]
            beta = pars[i+2]
            # loga_cdm = pars[i+3]
            # b_cdm = pars[i+4]
            # logc_cdm =pars[i+5]
            newmf = mf.WDM_stream(m_wdm=m_wdm,gamma=gamma, beta=beta)#, loga_cdm=loga_cdm,b_cdm=b_cdm,logc_cdm=logc_cdm)
        elif mftype == 'Press Schechter':
            del_crit = pars[i+0]
            newmf = mf.PressSchechter_test(del_crit = del_crit)
        elif mftype == 'PBH':
            logf_pbh = pars[i+0]
            newmf = mf.PBH(logf_pbh = logf_pbh)
        else:
           raise NotImplementedError("""Need to add this mass function to
           sampler.""")

        newmp = []
        n_lens = sum(newmf.n_l.astype(int))
        if mptype == 'ps':
            if n_lens == 1:
                index = np.nonzero(newmf.n_l.astype(int))
                kwargs['Ml'] = int(newmf.m_l[index[0]]) * u.Msun
                newmp = mp.PointSource(**kwargs)
            else:
                for newmf_ml, num_lenses in zip(newmf.m_l, newmf.n_l.astype(int)):
                    kwargs['Ml'] = newmf_ml*u.Msun
                    newmp.extend([mp.PointSource(**kwargs)for _ in range(num_lenses)])
                if len(newmp) == 0:
                    kwargs['Ml'] = int(newmf.m_l[2]) * u.Msun
                    newmp = mp.PointSource(**kwargs)

        elif mptype == 'gaussian':
            kwargs['R0'] = 10**pars[0]*u.pc
            if n_lens ==1:
                index = np.nonzero(newmf.n_l)
                kwargs['Ml'] = newmf.m_l[index[0]] * u.Msun
                newmp = mp.Gaussian(**kwargs)
            else:
                for newmf_ml, num_lenses in zip(newmf.m_l, newmf.n_l):
                    kwargs['Ml'] = newmf_ml * u.Msun
                    newmp.extend([mp.Gaussian(**kwargs) for _ in range(num_lenses)])
                if len(newmp) == 0:
                    kwargs['Ml'] = newmf.m_l[0] * u.Msun
                    newmp = mp.Gaussian(**kwargs)
        elif mptype == 'nfw':
            kwargs['c200'] = 10**pars[0]
            #print('in nfw makenewmass')
            if n_lens == 1: ##Case with only one lens, add the mass profile corresponding to it as an object not list
                index = np.nonzero(newmf.n_l.astype(int))
                kwargs['Ml'] = int(newmf.m_l[index[0]]) * u.Msun
                newmp = mp.NFW(**kwargs)
            else:
                #print(newmf.n_l)
                for ind123, (newmf_ml, num_lenses) in enumerate(zip(newmf.m_l, newmf.n_l.astype(int))):
                    #print(ind123)
                    kwargs['Ml'] = newmf_ml * u.Msun
                    newmp.extend([mp.NFW(**kwargs) for _ in range(num_lenses)])
                if len(newmp) == 0: ##Case where there are no lens, assume 1 exists in the lowest mass bin
                    kwargs['Ml'] = int(newmf.m_l[2]) * u.Msun
                    newmp = mp.NFW(**kwargs)

        else:
            raise NotImplementedError("""Need to add this mass profile/mass function to
            sampler.""")
            # print(type(newmp[0]), np.size(newmp))
            #print('done w makenewmass')
        return newmp, newmf #Array of mp