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
import cProfile

seed = 0

class AccelData():

    def __init__(self, survey,nstars=1000, ndims=1, wdm=False):
        rng = np.random.default_rng(seed=seed)
        if wdm:
            cdm = mf.CDM_Test()
            nfw = mp.NFW(Ml=1.e5*u.Msun, c200= 13)
            self.cdm_alphal = self.samplealphal([np.log10(13),np.log10(3.26 * 10 ** -5),-1.9,np.log10(2.57 * 10 ** 7)],
                                                survey,nstars,mptype=nfw, mftype=cdm)
            data = pd.DataFrame(self.cdm_alphal.value,columns=['a_x', 'a_y'])
            print('CDM data type:',type(self.cdm_alphal))
                #sampler.make_new_mass([2.5,np.log10(3.26 * 10 ** -5),-1.9,np.log10(2.57 * 10 ** 7)], 'CDM','nfw')
        else:
            if ndims == 2:
                data = pd.DataFrame(rng.standard_normal(size=(nstars,ndims))*survey.alphasigma.value,
                        columns=['a_x', 'a_y'])
                # path = STARDATADIR+'Roman_3_2_0_CDMTest.dat'
                # data = pd.read_csv(path)
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
        #print(np.shape(bb))
        self.alphal = self.samplealphal(pars,survey,nstars,mptype=mptype,mftype=mftype)
        #FIXME
        # if self.massprofile.type == 'ps':
        #     # if ps, need to do snr check and re-sample if any have too high snr. this stops walkers from getting too stuck.
        #      alphal = self.snr_check(alphal, pars)
        #      if alphal == "error":
        #          return -np.inf
        if np.any(np.isnan(self.alphal)): #FIXME Should not be needed!
            # print('inside np.isnan lnlike')
            return -np.inf
        try:
            # print('Inside try, alphal.value',alphal.value)
            # print('Inside try, data', self.data.to_numpy())
            diff = self.alphal.value - self.data.to_numpy()
            # print('diff', diff)
        except:
            return -np.inf
        chisq = -0.5 * np.sum((diff)**2 / survey.alphasigma.value**2 -np.log(2 * np.pi * survey.alphasigma.value**2))
        return chisq

    def samplealphal(self, pars, survey, nstars,mptype=None,mftype=None):
        rv = gsh.initialize_dist(target=survey.target,
                                 rmax=survey.maxdlens.to(u.kpc).value)
        self.rdist = rv
        #print('stars',nstars)
        self.bs = np.logspace(-8, np.log10(np.sqrt(2)*survey.fov_rad),nstars)
        #print('bs min',np.min(self.bs))
        newmp, newmassfunction = self.make_new_mass(pars,mptype,mftype)
        if np.size(newmp) > 1:
            mp_indices=np.random.randint(0, len(newmp), nstars)
            newmassprofile = [newmp[i] for i in mp_indices]
        else:
            newmassprofile = newmp
        nlens = newmassfunction.n_l
        if sum(nlens) ==0:
            nlens[0] = 1
        priorpdf = pdf(self.bs, a1=survey.fov_rad, a2=survey.fov_rad,
            n=sum(nlens))
        #print(priorpdf, self.bs, sum(nlens))
        if np.any(np.isnan(priorpdf)):
            #print('inside np.isnan')
            return -np.inf

        priorpdfspline = UnivariateSpline(np.log10(self.bs[priorpdf>0]),
                np.log10(priorpdf[priorpdf>0]), ext='zeros', s=0)
        self.dists = self.rdist.rvs(nstars) * u.kpc
        x = np.log10(self.bs)
        self.y = 10**priorpdfspline(np.log10(self.bs))
        self.sci_s = scipy.interpolate.interp1d(x, self.y, fill_value='extrapolate')
        sci = self.sci_s(x)
        #print('pdf:',sci/sum(sci))
        temp = np.random.choice(x, nstars, p=sci / sum(sci)) #* self.dists
        self.beff = 10**(np.ones(temp.shape)*np.average(temp)) * self.dists
        # self.beff = 10 ** (temp) * self.dists
        #self.beff = 10 ** (np.random.choice(x, nstars, p=sci / sum(sci))) * self.dists
        #print('dist min', np.min(self.dists))
        self.vl = scipy.stats.truncnorm.rvs(a=0, b=550./220, loc=0.,scale=220, size=nstars)
        self.bvec = np.zeros((nstars, 2))
        self.vvec = np.zeros((nstars, 2))
        btheta = np.random.rand(nstars)* 2. * np.pi
        vtheta = np.random.rand(nstars)* 2. * np.pi
        self.bvec[:, 0] = self.beff * np.cos(btheta)
        self.bvec[:, 1] = self.beff * np.sin(btheta)
        self.vvec[:, 0] = self.vl * np.cos(vtheta)
        self.vvec[:, 1] = self.vl * np.sin(vtheta)

        self.bvec *= u.kpc
        self.vvec *= u.km / u.s
        ## get alphal given sampled other params.
        #start = time.perf_counter()
        # print('bvec:',bvec)
        # print('vvec:', vvec)
        self.alphal = lm.alphal(newmassprofile, self.bvec, self.vvec)
        # if bool:
        #     bb.append(np.linalg.norm(self.bvec, axis=1))
        #     #print('beff vs bvec',np.isclose(np.linalg.norm(self.bvec, axis=1),self.beff,atol=0))
        #     aa.append(self.alphal)
        #end = time.perf_counter()
        #print(f'Time taken for lensing model: {(end-start):.6f} second')
        return self.alphal

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
            newmf = mf.PowerLaw(m_l=mft.m_l,logM_0=logM0, logalpha=logalpha)
        elif mftype == 'Tinker':
            # A = pars[i+0]
            a = pars[i+0]
            b = pars[i+1]
            c = pars[i+2]
            #k_b = pars[i+4]
            #n_b = pars[i+5]
            #k_s = pars[i+6]
            newmf = mf.Tinker(m_l=mft.m_l, a= a, b= b, c= c)#, k_b=k_b, n_b=n_b, k_s=k_s)
        elif mftype == 'CDM':
            # loga = pars[i+0]
            b = pars[i+0]
            logc = pars[i+1]
            newmf = mf.CDM_Test(m_l=mft.m_l, b = b,logc = logc)
        elif mftype == 'WDM Lensing':
            mwdm = pars[i+0]
            beta = pars[i+1]
            newmf = mf.WDM_lensing(m_l=mft.m_l,mwdm=mwdm, beta=beta)
        elif mftype == 'WDM Stream':
            logmwdm = pars[i+0]
            gamma = pars[i+1]
            beta = pars[i+2]
            # loga_cdm = pars[i+3]
            # b_cdm = pars[i+4]
            # logc_cdm =pars[i+5]
            newmf = mf.WDM_stream(m_l=mft.m_l,logmwdm=logmwdm,gamma=gamma, beta=beta)#, loga_cdm=loga_cdm,b_cdm=b_cdm,logc_cdm=logc_cdm)
        elif mftype == 'Press Schechter':
            del_crit = pars[i+0]
            newmf = mf.PressSchechter_test(m_l=mft.m_l,del_crit = del_crit)
        elif mftype == 'PBH':
            logf_pbh = pars[i+0]
            newmf = mf.PBH(m_l=mft.m_l,logf_pbh = logf_pbh)
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
                    print('no lens sampler')
                    kwargs['Ml'] = int(newmf.m_l[0]) * u.Msun
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
                    print('no lens sampler')
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
                for newmf_ml, num_lenses in zip(newmf.m_l, newmf.n_l.astype(int)):
                    #print(ind123)
                    kwargs['Ml'] = newmf_ml * u.Msun
                    newmp.extend([mp.NFW(**kwargs) for _ in range(num_lenses)])
                if len(newmp) == 0: ##Case where there are no lens, assume 1 exists in the lowest mass bin
                    print('no lens sampler')
                    kwargs['Ml'] = int(newmf.m_l[0]) * u.Msun
                    newmp = mp.NFW(**kwargs)

        else:
            raise NotImplementedError("""Need to add this mass profile/mass function to
            sampler.""")
            # print(type(newmp[0]), np.size(newmp))
            #print('done w makenewmass')
        return newmp, newmf #Array of mp