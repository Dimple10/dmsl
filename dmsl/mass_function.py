'''
Defines mass functions
'''
import astropy.cosmology
from astropy.cosmology import WMAP7 as cosmo
import astropy.units as u
import numpy as np
import pickle as pkl
import dmsl.galaxy_subhalo_dist as gsh
from dataclasses import dataclass,field
import dmsl.survey as surv
from scipy.integrate import quad_vec
from collections import Counter
from classy import Class
import scipy
import random
import time

RHO_CRIT = cosmo.critical_density(0.)
Rho_mean = cosmo.Om0 * RHO_CRIT
Rho_dm = gsh.density(8. * u.kpc)
# Rho_dm = 7.23e10*(u.Msun/u.kpc**3)
MW_vol = 4/3*np.pi*(100*u.kpc)**3 #Milky Way vol approx. as a sphere
# MW_vol = np.pi*(15*u.kpc)**2 * 3*u.kpc #MW as a disk (not halo just galaxy)
h = cosmo.h

@dataclass #(example in survey.py)
class MassFunction:
    n_l : list = field(default_factory = list)
    m_l: list = field(default_factory = list)
    sur: surv = surv.Roman()

    @property
    def check_density(self):
        self.vol = self.sur.fov_rad ** 2 * self.sur.maxdlens**3 / 3.
        m_dm = (Rho_dm * vol).to(u.Msun)
        DM = m_dm.to_value()
        sum = np.sum(self.n_l*self.m_l)
        if (int(sum) in range(int(0.9*DM), int(1.1*DM))): #Within 10% of DM Mass
            return True
        else:
            return False

    def __post_init__(self):
        if(not self.check_density()):
            raise RuntimeWarning("Check density failed! Retry with different mass values")
        else:
            pass

    def update(self):
        self.__post_init__()

@dataclass
class PowerLaw(MassFunction):
    Name: str = 'PowerLaw'
    m_l : list = field(default_factory=lambda:np.logspace(0,2,100))
    n_l: list = field(default_factory=lambda: np.zeros((100)))  ##FIXME len(m_l)
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    logalpha: int = 0
    logM_0: float = 0
    nparams: int = 2
    param_names: list =  field(default_factory=lambda:['logalpha', 'logM_0'])
    param_range: dict = field(default_factory=lambda:{'logalpha':(-3,5), 'logM_0':(-5,7)})

    def find_Nl(self):
        #Calculating the normalized number of lensess
        for i in range(len(self.m_l)):
            self.den_n_l[i] = np.ceil((self.m_l[i] / 10**self.logM_0) ** (10**self.logalpha))

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3. #* 12 * 8 * 10
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = np.insert(integr, 0, 0)
        N = np.diff(integr, prepend=0)
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        N = norm * N
        nlens = sum(N)
        ran_samp = np.random.choice(self.m_l, np.int64(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]

        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0, len(self.n_l)-1)] = 1

    def __post_init__(self):
        self.find_Nl()

@dataclass
class Tinker(MassFunction):
    Name: str = 'Tinker'
    m_l: list = field(default_factory=lambda: np.logspace(4, 12, 7))
    den_n_l: list = field(default_factory=lambda: np.zeros(100))
    n_l: list = field(default_factory=lambda: np.zeros(100))
    A: float = 0.260  #FIXME Used values from colossus (https://bitbucket.org/bdiemer/colossus/src/master/colossus/lss/mass_function.py)
    a: float = 2.66 #power
    b: float = 1.41
    c: float = 2.44 #power
    sig: list = field(default_factory=lambda: [1 for i in range(100)])
    der: list = field(default_factory=lambda: [1 for i in range(100)])
    R: list = field(default_factory=lambda: [1 for i in range(100)])
    f: list = field(default_factory=lambda: [1 for i in range(100)])
    A_s: float = 2.105 * 10 ** -9
    n_s: float = 0.9665
    logk_b: float = (np.log10(13) * (1 / u.Mpc))  # Units of Mpc^-1
    n_b: float = 2.0  # or 3.0 (Fig #7 in Power of Halometry)
    k_s: float = 0.05 * (1 / u.Mpc)  # Units of Mpc^-1
    #cosmo:astropy.cosmology.Cosmology() = cosmo
    nparams: int = 2
    param_names: list = field(default_factory=lambda:['logk_b', 'n_b'])#, 'k_s'])#['a', 'b', 'c'])
    param_range: dict = field(default_factory=lambda:{'logk_b': (-2,2), 'n_b':(1,4)}) #{ 'a': (1.8, 5), 'b':(0.001, 100), 'c':(1.8, 5)})

    def getPk(self):
        #start=time.time()
        cosmol = Class()
        cosmol.set({'P_k_max_1/Mpc':100,'omega_b':cosmo.Ob0*h**2,'omega_m':cosmo.Om0*h**2,'h':cosmo.h,'A_s':2.100549e-09,
                    'n_s':0.9660499,'tau_reio':0.05430842})
        cosmol.set({'output':'mPk,mTk'})
        cosmol.compute()
        phi_k = cosmol.get_primordial()['k [1/Mpc]']
        phi_class = cosmol.get_primordial()['P_scalar(k)']
        kk = np.logspace(-3, np.log10(100), 10000)  # k in 1/Mpc
        Pk = [cosmol.pk(s,0.)*h**3 for s in kk]  # P(k) in (Mpc)**3
        self.phi_class_interp = scipy.interpolate.interp1d(np.log10(phi_k), np.log10(phi_class), fill_value='extrapolate')
        self.F = scipy.interpolate.interp1d(np.log10(kk), np.log10(Pk), fill_value='extrapolate')
        # print('Got throught getPk')
        return self.F

    def phi(self,k):
        # print('1:',10**((self.logk_b*u.Mpc).to('')))
        # print('in phi')
        # print(self.logk_b)
        return np.piecewise(k, [k<10**((self.logk_b*u.Mpc).to('')), k>=10**((self.logk_b*u.Mpc).to(''))],
               [lambda k:(self.A_s * (k / ((u.Mpc * self.k_s).to(''))) ** (self.n_s - 1)).to(''),
                lambda k:(self.A_s * (10**((self.logk_b*u.Mpc).to('')) / ((self.k_s*u.Mpc).to(''))) ** (self.n_s - 1) * (k / (10**((self.logk_b*u.Mpc).to('')))) ** (self.n_b - 1)).to('')])

    def calc_f(self):
        self.f = (self.A * ((np.array(self.sig) / self.b) ** (-1 * self.a) + 1)) * np.exp(
            (-1 * self.c / (np.array((self.sig)) ** 2)))
        # print('Got throught calc_f')
       # print("f(sig)= ", self.f)

    def radius(self):
        self.R = (3*self.m_l * u.M_sun/ (4 * np.pi * (Rho_mean.to(u.M_sun/u.pc**3)))) ** (1 / 3)
        #print("R= ", self.R)
        # print('Got throught radius')
        return self.R

    def func(self,k,i):
        f_1 = k**2/(2*np.pi**2) #1/k  #In units of u.Mpc
        f_2 = 3 / ((k * (1 / u.Mpc) * self.R[i]).to('') ** 3)#* self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad) - (k * (1 / u.Mpc) * self.R[i]).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f = (np.abs(f_2 * f_3) ** 2 * f_1).to('')
        # print('Got throught func')
        return f #* self.phi(k)

    def calc_sig(self):
        k = np.logspace(-3, 5, 100)
        # print(10 ** self.F(np.log10(k)))
        self.transfer = 10 ** self.F(np.log10(k)) / 10 ** self.phi_class_interp(np.log10(k))
        # print('After transfer in calc_sig')
        # print(10 ** self.phi_class_interp(np.log10(k)))
        self.Pk = (self.phi(k) * self.transfer)
        # print('after newPk in calc_sig')
        int_val = []
        for i in range(len(self.R)):
            integrand = self.func(k, i) * self.Pk
            integr = scipy.integrate.trapezoid(integrand, k)
            int_val.append(integr)
        for i in range(len(self.sig)):
            self.sig[i] = np.sqrt(int_val[i].value)
        # F = self.getPk()
        # D2 = lambda k: 10**F(np.log10(k))*k**3/(2*np.pi**2*self.phi(k))
        # integrand = lambda k: self.func(k)* 10**F(np.log10(k))#D2(k)**2 *self.phi(k)
        # f_1 = quad_vec(integrand, 1e-3, 100000, limit=100)
        # for i in range(len(self.sig)):
        #     self.sig[i] = np.sqrt(f_1[0][i]).to('')
        #self.sig = f_1[0]
        #print("Sig: ", self.sig)
        return self.sig

    def func_der(self,k,i):
        f_1 = k**2/(2*np.pi**2) * u.Mpc #1/k *u.Mpc #
        f_2 =  3 / ((k * (1 / u.Mpc) * self.R[i]).to('') ** 3)# * self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad) - (k * (1 / u.Mpc) * self.R[i]).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_4 = (-3 / self.R[i].value) * 1/u.Mpc *(u.Mpc / u.pc).to('') * np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_5 = 3 * (k * (1 / u.Mpc)) * np.cos((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_6 = 1/u.Mpc * (k ** 2 * (1 / u.Mpc) * self.R[i]).to('') * np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f = ((f_2 * np.abs(f_3)) * f_1 * f_2 * (f_4 + f_5 + f_6)).to('')
        # print('Got throught func_der')
        return f #* self.phi(k)

    def der_sig(self):
        #F = self.getPk()
        # D2 = lambda k: 10 ** F(np.log10(k)) * k ** 3 / (2 * np.pi ** 2 * self.phi(k))
        # integrand_der = lambda k: self.func_der(k) * 10**F(np.log10(k)) #D2(k)**2 *self.phi(k)
        # f_1 = quad_vec(integrand_der, 1e-3, 100000, limit=100)
        # self.der = f_1[0]
        #print("dsig/dR = ", self.der/self.sig)
        k = np.logspace(-3, 5, 100)
        int_val = []
        for i in range(len(self.R)):
            integrand = self.func_der(k, i)*self.Pk
            integr = scipy.integrate.trapezoid(integrand, k)
            int_val.append(integr.value)
        self.der = int_val
        # print('Got throught der_sig')
        return self.der #units of 1/u.Mpc

    def find_Nl(self):
        for i in range(len(self.n_l)): ##n_l units of 1/Mpc**3
            self.den_n_l[i] = (4 / 3 * np.pi * (Rho_mean.to(u.M_sun/u.Mpc**3).value)) ** (-1 / 3) * 1/3 * (self.m_l[i]) ** (-2 / 3)* \
                              self.f[i] * (Rho_mean.to(u.M_sun/u.Mpc**3).value) / self.m_l[i] * (-self.der[i]/self.sig[i]**2) #*self.m_l[i]

        # self.den_n_l*=MW_vol.to(u.Mpc**3).value
        # self.den_n_l *= MW_vol.value
        #Calculating the normalization
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3. #* 12 * 8 * 10
        correction = (vol / MW_vol).value * 76.66
        # self.den_n_l *= (vol.to(u.Mpc**3)).value * 76.66
        # print((vol.to(u.Mpc**3)).value)
        integr = scipy.integrate.cumulative_trapezoid((self.den_n_l*correction), self.m_l)
        nlens = integr[-1]
        # integr = np.insert(integr, 0, 0)
        # N = np.diff(integr, prepend=0)
        # This is dn/dM not dN/dM as everything else so multiplying by vol already!
        # N *= vol.value
        # N *=MW_vol.to(u.Mpc**3).value
        # m_dm = np.sum(N * self.m_l) * u.Msun
        # m_sur = Rho_dm * vol
        # norm = m_sur / m_dm
        # print('Tinker norm=', norm)
        # norm = vol/MW_vol
        # N = norm * N
        # nlens = sum(N)#/10**5
        # print('lens',nlens)
        ran_samp = np.random.choice(self.m_l, np.int64(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1
        # print('Got throught find_nl')

    def __post_init__(self):
        self.radius()
        self.getPk()
        self.calc_sig()
        self.calc_f()
        self.der_sig()
        self.find_Nl()
        pass

@dataclass
class Tinker_Mishra(MassFunction):
    Name: str = 'Tinker'
    m_l: list = field(default_factory=lambda: np.logspace(4, 12, 7))
    den_n_l: list = field(default_factory=lambda: np.zeros(100))
    n_l: list = field(default_factory=lambda: np.zeros(100))
    A: float = 0.260  #FIXME Used values from colossus (https://bitbucket.org/bdiemer/colossus/src/master/colossus/lss/mass_function.py)
    a: float = 2.66 #power
    b: float = 1.41
    c: float = 2.44 #power
    sig: list = field(default_factory=lambda: [1 for i in range(100)])
    der: list = field(default_factory=lambda: [1 for i in range(100)])
    R: list = field(default_factory=lambda: [1 for i in range(100)])
    f: list = field(default_factory=lambda: [1 for i in range(100)])
    A_s: float = 2.105 * 10 ** -9
    n_s: float = 0.9665
    logkb: float = (np.log10(13) * (1 / u.Mpc))  # Units of Mpc^-1
    nb: float = 3.0  # or 3.0 (Fig #7 in Power of Halometry)
    k_s: float = 0.05 * (1 / u.Mpc)  # Units of Mpc^-1
    nparams: int = 2
    param_names: list = field(default_factory=lambda:['logkb', 'nb'])#, 'k_s'])#['a', 'b', 'c'])
    param_range: dict = field(default_factory=lambda:{'logkb': (np.log10(5),np.log10(50)), 'nb':(1,3)}) #{ 'a': (1.8, 5), 'b':(0.001, 100), 'c':(1.8, 5)})

    # def getPk(self):
    #     cosmol = Class()
    #     cosmol.set({'P_k_max_1/Mpc':100,'omega_b':cosmo.Ob0*h**2,'omega_m':cosmo.Om0*h**2,'h':cosmo.h,'A_s':2.100549e-09,
    #                 'n_s':0.9660499,'tau_reio':0.05430842})
    #     cosmol.set({'output':'mPk,mTk'})
    #     cosmol.compute()
    #     phi_k = cosmol.get_primordial()['k [1/Mpc]']
    #     phi_class = cosmol.get_primordial()['P_scalar(k)']
    #     kk = np.logspace(-3, np.log10(100), 10000)  # k in 1/Mpc
    #     Pk = [cosmol.pk(s,0.)*h**3 for s in kk]  # P(k) in (Mpc)**3
    #     self.phi_class_interp = scipy.interpolate.interp1d(np.log10(phi_k), np.log10(phi_class), fill_value='extrapolate')
    #     self.F = scipy.interpolate.interp1d(np.log10(kk), np.log10(Pk), fill_value='extrapolate')
    #     return self.F
    def getPk(self):
        with open('tinker_interpFunc.pkl', 'rb') as file:
            self.interp_func = pkl.load(file)
        # print('Past getPk')
        return self.interp_func

    def phi(self,k):
        return np.piecewise(k, [k<10**((self.logkb*u.Mpc).to('')), k>=10**((self.logkb*u.Mpc).to(''))],
               [lambda k:(self.A_s * (k / ((u.Mpc * self.k_s).to(''))) ** (self.n_s - 1)).to(''),
                lambda k:(self.A_s * (10**((self.logkb*u.Mpc).to('')) / ((self.k_s*u.Mpc).to(''))) ** (self.n_s - 1) * (k / (10**((self.logkb*u.Mpc).to('')))) ** (self.nb - 1)).to('')])

    def calc_f(self):
        self.f = (self.A * ((np.array(self.sig) / self.b) ** (-1 * self.a) + 1)) * np.exp(
            (-1 * self.c / (np.array((self.sig)) ** 2)))
        # print('Past calc_f')

    def radius(self):
        self.R = (3*self.m_l * u.M_sun/ (4 * np.pi * (Rho_mean.to(u.M_sun/u.pc**3)))) ** (1 / 3)
        # print('Past radius')
        return self.R

    def func(self,k,i):
        f_1 = k**2/(2*np.pi**2) #1/k  #In units of u.Mpc
        f_2 = 3 / ((k * (1 / u.Mpc) * self.R[i]).to('') ** 3)#* self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad) - (k * (1 / u.Mpc) * self.R[i]).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f = (np.abs(f_2 * f_3) ** 2 * f_1).to('')
        # print('Past func')
        return f #* self.phi(k)

    def calc_sig(self):
        k = np.logspace(-3, 5, 1000)
        # self.transfer = 10 ** self.F(np.log10(k)) / 10 ** self.phi_class_interp(np.log10(k))
        # self.Pk = (self.phi(k) * self.transfer)
        self.Pk = 10**self.interp_func([[10**(self.logkb.value),self.nb,np.log10(kk)] for kk in k])
        int_val = []
        for i in range(len(self.R)):
            integrand = self.func(k, i) * self.Pk
            integr = scipy.integrate.trapezoid(integrand, k)
            int_val.append(integr)
        for i in range(len(self.sig)):
            self.sig[i] = np.sqrt(int_val[i].value)
        # print('Past calc_sig')
        return self.sig

    def func_der(self,k,i):
        f_1 = k**2/(2*np.pi**2) * u.Mpc #1/k *u.Mpc #
        f_2 =  3 / ((k * (1 / u.Mpc) * self.R[i]).to('') ** 3)# * self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad) - (k * (1 / u.Mpc) * self.R[i]).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_4 = (-3 / self.R[i].value) * 1/u.Mpc *(u.Mpc / u.pc).to('') * np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_5 = 3 * (k * (1 / u.Mpc)) * np.cos((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_6 = 1/u.Mpc * (k ** 2 * (1 / u.Mpc) * self.R[i]).to('') * np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f = ((f_2 * np.abs(f_3)) * f_1 * f_2 * (f_4 + f_5 + f_6)).to('')
        # print('Past func_der')
        return f #* self.phi(k)

    def der_sig(self):
        k = np.logspace(-3, 5, 1000)
        int_val = []
        for i in range(len(self.R)):
            integrand = self.func_der(k, i)* self.Pk
            integr = scipy.integrate.trapezoid(integrand, k)
            int_val.append(integr.value)
        self.der = int_val
        # print('Past der_sig')
        return self.der #units of 1/u.Mpc

    def find_Nl(self):
        for i in range(len(self.n_l)): ##n_l units of 1/Mpc**3
            self.den_n_l[i] = (4 / 3 * np.pi * (Rho_mean.to(u.M_sun/u.Mpc**3).value)) ** (-1 / 3) * 1/3 * (self.m_l[i]) ** (-2 / 3)* \
                              self.f[i] * (Rho_mean.to(u.M_sun/u.Mpc**3).value) / self.m_l[i] * (-self.der[i]/self.sig[i]**2) #*self.m_l[i]

        # print('Past den_n_l in find_Nl')
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3. #* 12 * 8 * 10
        correction = (vol / MW_vol).value * 76.66
        # correction = vol.value
        # correction = 1
        integr = scipy.integrate.cumulative_trapezoid((self.den_n_l*correction), self.m_l)
        nlens = integr[-1]
        ran_samp = np.random.choice(self.m_l, np.int64(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1

    def __post_init__(self):
        self.radius()
        self.getPk()
        self.calc_sig()
        self.calc_f()
        self.der_sig()
        self.find_Nl()
        pass

@dataclass
class CDM(MassFunction):
    Name: str = 'CDM'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 10))
    den_n_l: list = field(default_factory=lambda: np.zeros((10)))
    n_l: list = field(default_factory=lambda: np.zeros((10)))
    loga: float = np.log10(3.26*10**-5)
    b: float = -1.9
    logc: float =np.log10(2.57*10**7)
    nparams: int = 3
    param_names: list = field(default_factory=lambda:['loga', 'b', 'logc'])
    param_range: dict = field(default_factory=lambda: {'loga':(-8, 3), 'b': (-4, -1), 'logc':(3, 9)})

    def find_Nl(self):
        self.den_n_l = (10**self.loga) * ((self.m_l/(10**self.logc))** self.b) ##Units of M_sun^-1
        ##FIXME Higher by a factor of 10 than in the Shutz paper

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3. #*12 *8 *10
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l,self.m_l)
        nlens=sum(np.diff(integr))
        #print(nlens)
        # Random sampling from m_l, n_lens times with probablity dn/dm
        ran_samp = np.random.choice(self.m_l, np.int64(nlens), p=self.den_n_l / sum(self.den_n_l))
        print('ran_samp',np.shape(ran_samp))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        print('nl=',nl)
        #Counting number of lenses per mass bin
        # hist, _ = np.histogram(ran_samp, bins=self.m_l)
        # hist = np.append(hist,0)
        # nl=np.diff(integr,prepend=integr[0])
        # nl.append(0)
        m_dm = np.sum(nl * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        if m_dm != 0:
            norm = m_sur / m_dm
            l = nl * norm
            print('l',l)
        else:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1
            return 0

        #Poisson sampling to get final number of lenses per mass bin
        self.n_l = np.random.poisson(l)
        if sum(self.n_l)==0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1
        print('final nl=',self.n_l)

    def __post_init__(self):
        self.find_Nl()

@dataclass
class LCDM_Tinker(MassFunction):
    Name: str = 'LCDM'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 100))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    gamma : float = 1.9
    nparams: int = 1
    param_names: list = field(default_factory=lambda:['gamma'])
    param_range: dict = field(default_factory=lambda: {'gamma':(1,3)})

    def find_Nl(self):
        self.den_n_l = self.m_l **(-self.gamma)
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.  # * 12 * 8 * 10
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = np.insert(integr, 0, 0)
        N = np.array(np.diff(integr, prepend=0))
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        # norm = m_sur/m_dm
        norm = vol / MW_vol
        # print('CDM norm=',norm)
        # N = norm*N
        # print('N after norm=', sum(N))
        nlens = sum(N)  # /10**10 #Lowered by 10**5 only for MW volume
        if nlens > 1:
            print('nlens>1 in LCDM,', int(nlens))
        # print('nlens,norm',nlens,norm)
        ran_samp = np.random.choice(self.m_l, int(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        # print('nl=', nl)

        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0, len(self.n_l) - 1)] = 1

    def __post_init__(self):
        self.find_Nl()

@dataclass
class CDM_Test(MassFunction):
    Name: str = 'CDM'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 100))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    loga: float = np.log10(3.26 * 10 ** -5)
    b: float = -1.9
    logc: float = np.log10(2.57 * 10 ** 7)
    nparams: int = 2
    param_names: list = field(default_factory=lambda: ['b', 'logc'])
    param_range: dict = field(default_factory=lambda: {'b': (-2.5, -1), 'logc': (6, 8)})

    def find_Nl(self):
        self.den_n_l = (10 ** self.loga) * ((self.m_l / (10 ** self.logc)) ** self.b)  ##Units of M_sun^-1
        ##FIXME Higher by a factor of 10 than in the Shutz paper
        # print('dn/dm=',self.den_n_l)
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3. #* 12 * 8 * 10
        # print('vol=',vol)
        ##For Roman vol Only
        ## Since its dN/dM, dividing by MW vol and multiplying by roman vol
        # correction= (vol/MW_vol).value*76.66#Correction factor w/ avg density in roman vs MW
        correction = 1
        #Should be in Mpc but converting both to Mpc will cancel out
        # print((vol/MW_vol).value)

        self.integr = scipy.integrate.cumulative_trapezoid((self.den_n_l*correction), self.m_l)
        nlens = self.integr[-1]
        print('MW',nlens)
        # print(correction)
        # print('Last in cumulative trapz:', self.integr[-1])
        # self.integr = np.insert(self.integr,0,0)
        # print('integr + size:', integr, np.size(integr))
        # self.N = np.array(np.diff(self.integr,prepend=0))
        # print('N tot before norm', sum(N))
        # m_dm = np.sum(self.N * self.m_l) * u.Msun
        # m_sur = Rho_dm * vol
        # # norm = m_sur/m_dm
        # norm = vol / MW_vol
        # print('CDM norm=',norm)
        # N = norm*N
        # print('N after norm=', sum(N))
        # nlens= sum(self.N)#/10**5 #Lowered by 10**5 only for MW volume Lowered for WDM_stream calc coz irrelevant!
        # print('nlens',nlens)
        # if nlens>1000:
        #     print('nlens>1000 in CDM,', int(nlens))
        # print('nlens,norm',nlens,norm)
        if np.log10(nlens)>9:
            nlens = nlens/10**(np.floor(np.log10(nlens))-8)
            print('lowered nlens:',nlens)
        ran_samp = np.random.choice(self.m_l, int(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        # print('nl=', nl)

        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1
        #print('final nl=', self.n_l)

    def __post_init__(self):
        self.find_Nl()

@dataclass
class WDM_stream(MassFunction):
    Name: str = 'WDM Stream'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 100))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    M_hm: float = 1.0
    gamma: float = 2.7
    beta: float = 0.99
    logmwdm: float = np.log10(6.3) #in KeV
    omega_wdm: float = 0.25
    loga_cdm: float = np.log10(3.26*10**-5)
    b_cdm: float = -1.9
    logc_cdm: float = np.log10(2.57*10**7)
    a : float = 1.65*10**10
    b: float = -3.33
    c :float = 1.33
    d :float = 2.66
    nparams: int = 3 #m_wdm, gamma, beta
    param_names: list = field(default_factory=lambda:['logmwdm', 'gamma', 'beta'])#, 'loga_cdm','b_cdm','logc_cdm'])
    param_range: dict = field(default_factory=lambda:{'logmwdm': (-2,4),'gamma': (0.01, 60), 'beta':(0.1,3)})#,
                                                     # 'loga_cdm':(-8, 2), 'b_cdm': (-4, -0.01), 'logc_cdm':(4, 10)})
    def calc_Mhm(self):
        self.M_hm = self.a * (10**self.logmwdm)** self.b * (self.omega_wdm/0.25)**self.c #* (h/0.7)**2.66
        #print(self.M_hm)

    def find_Nl(self):
        cdm = CDM_Test(m_l=self.m_l, sur=self.sur)#, loga=self.loga_cdm, b=self.b_cdm, logc=self.logc_cdm) #return dN/dM, we need dN/dlnM
        cdm_den_nl = cdm.den_n_l * cdm.m_l #CDM dn/dm without correction
        #print('cdm den:',cdm_den_nl)
        self.den_n_l = (1 + self.gamma*self.M_hm/self.m_l)**(-1*self.beta) * cdm_den_nl
        #Convert from dN/dlnM to dN/dM--
        self.den_n_l = self.den_n_l / self.m_l

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.  # * 12 * 8 * 10l
        correction = (vol / MW_vol).value * 76.66  # Correction factor w/ avg density in roman vs MW
        self.integr = scipy.integrate.cumulative_trapezoid((self.den_n_l * correction), self.m_l)
        nlens = self.integr[-1]
        # N = np.diff(integr, prepend=0)
        # m_dm = np.sum(N * self.m_l) * u.Msun
        # m_sur = Rho_dm * vol
        # # norm = m_sur / m_dm
        # norm = vol / MW_vol
        # print('WDM norm=', norm)
        # N = norm * N
        # nlens = sum(N)#/10**5
        # print('nlens WDM',nlens)
        ran_samp = np.random.choice(self.m_l, int(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1
        # print('nl:',self.n_l)
        # integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l, initial=integr[0])
        # N = np.diff(integr, prepend=integr[0])
        # m_dm = np.sum(N * self.m_l) * u.Msun
        # m_sur = Rho_dm * vol
        # norm = m_sur / m_dm
        # self.n_l = norm * N

    def __post_init__(self):
        self.calc_Mhm()
        self.find_Nl()

@dataclass
class WDM_lensing(MassFunction):
    Name: str = 'WDM Lensing'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 100))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    M_hm: float = 1.0
    beta: float = 1.3 #Shutz reference
    mwdm: float = 6.3 #in KeV
    omega_wdm: float = cosmo.Odm0
    a: float = 1.65 * 10 ** 10
    b: float = -3.33
    c: float = 1.33
    d: float = 2.66
    nparams = 2
    param_names: list = field(default_factory=lambda:['mwdm','beta']) #FIXME To correct vals if needed
    param_range: dict = field(default_factory=lambda:{'mwdm':(0.01,500),'beta': (0.58, 8)})

    def calc_Mhm(self):
        self.M_hm = self.a * self.mwdm**self.b * (self.omega_wdm/0.25)**self.c * (h/0.7)**self.d
        #print(self.M_hm)

    def find_Nl(self):
        cdm = CDM_Test(m_l=self.m_l) #return dN/dM, we need dN/dlnM
        cdm_den_nl = cdm.den_n_l * cdm.m_l
        self.den_n_l = (1 + self.M_hm/self.m_l)**(-1*self.beta) * cdm_den_nl
        #Convert from dN/dlnM to dN/dM--
        self.den_n_l = self.den_n_l / self.m_l

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = np.insert(integr, 0, 0)
        N = np.diff(integr, prepend=0)
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        N = norm * N

        nlens = sum(N)
        ran_samp = np.random.choice(self.m_l, np.int64(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1

    def __post_init__(self):
        self.calc_Mhm()
        self.find_Nl()

@dataclass
class PBH(MassFunction): ##Check on normalization
    Name: str = 'PBH' ## DeRocco 2023 paper eq 6 + 7
    #m_l: list = field(default_factory=lambda: np.logspace(-8, -3, 10))
    m_l: list = field(default_factory=lambda: np.zeros((100)))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    sig: float = 1.0
    logfpbh: float = -2
    m_c: float = 1.0
    nparams: int = 1
    param_names: list = field(default_factory=lambda: ['logfpbh'])
    param_range: dict = field(default_factory=lambda: {'logfpbh': (-9, 0)})

    def calc(self):
        #print('m_l:',self.m_l)
        self.m_c = np.mean(np.log(self.m_l)) #FIXME see if you need np.log
        self.sig = np.std(np.log(self.m_l))
        #print('m_c,sig:',self.m_c,self.sig)
    def find_Nl(self):
        self.den_n_l = 10**self.logfpbh/(np.sqrt(2*np.pi) * 10**self.sig *self.m_l)*\
                       np.exp(-np.log(self.m_l/10**self.m_c)**2/(2*10**self.sig))  ##Units of M_sun^-1
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        rho_pbh = gsh.density(6*u.kpc)
        # self.den_n_l *= vol.value ##Assuming it's dn/dM
        # print('vol:',vol)
        # correction = 76.66
        # correction = 1
        # correction = (vol / MW_vol).value * 76.66
        # self.integr = scipy.integrate.cumulative_trapezoid((self.den_n_l * correction), self.m_l)
        self.integr = scipy.integrate.cumulative_trapezoid(((self.den_n_l * rho_pbh)/self.m_l), self.m_l)
        nlens = self.integr[-1]
        # integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        # integr = np.insert(integr,0,0)
        # N = np.diff(integr, prepend=0)
        # # print('N before norm',np.sum(N))
        # m_dm = np.sum(N* self.m_l) * u.Msun
        # m_sur = Rho_dm * vol
        # norm = m_sur/m_dm
        # # norm = vol / MW_vol
        # # print('PBH norm,total nlens=', norm, sum(norm*N))
        # # print('Milkyway norm, total nlens=', norm_MW, sum(norm_MW*N))
        # N = norm*N
        #print(m_dm, N)
        # nlens= sum(N)
        # print('PBH lens', nlens)
        ran_samp = np.random.choice(self.m_l, int(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        # print('nl,',nl)
        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0, len(self.n_l) - 1)] = 1

    def __post_init__(self):
        self.calc()
        self.find_Nl()

@dataclass
class PBH_Gaussian(MassFunction): ##Check on normalization
    Name: str = 'PBH'
    # m_l: list = field(default_factory=lambda: np.logspace(0, 3, 100))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    logmass: float = 0
    sigma: float = 10
    nparams: int = 2
    param_names: list = field(default_factory=lambda: ['logmass','sigma'])
    param_range: dict = field(default_factory=lambda: {'logmass': (0, 3), 'sigma':(0.072267, 67.39665)})

    def find_Nl(self):
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        dist = np.random.normal(loc=10**self.logmass, scale=self.sigma, size=1000)
        dist = dist[dist>=0]
        counts, bins = np.histogram(dist)
        self.m_l = bins[1:]
        m_dm = np.sum(counts) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = np.random.poisson(norm*counts)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0, len(self.n_l) - 1)] = 1

    def __post_init__(self):
        self.find_Nl()

@dataclass
class PressSchechter(MassFunction):
    Name: str = 'Press Schechter'
    m_l: list = field(default_factory=lambda: np.logspace(4, 12, 100))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    sig: list = field(default_factory=lambda: [1 for i in range(100)])
    der: list = field(default_factory=lambda: [1 for i in range(100)])
    R: list = field(default_factory=lambda: [1 for i in range(100)])
    del_crit: float = 1.686
    nparams: int = 1
    param_names: list = field(default_factory=lambda:['del_crit'])
    param_range: dict = field(default_factory=lambda:{'del_crit':(0.01,4)})

    def getPk(self,k): #Analytic fit from statsmodel to CLASS P(k)
        return 10**(1.5673598289074357 + np.log10(k)*-2.190072492256715 +np.log10(k)**2 *-0.46130280750744995)

    def radius(self):
        self.R = (3*self.m_l * u.M_sun/ (4 * np.pi * (Rho_mean.to(u.M_sun/u.pc**3)))) ** (1 / 3)
        return self.R

    def func(self,k):
        f_1 = 1/(2*np.pi**2) #1/k  #In units of u.Mpc
        f_2 = 3 / ((k * (1 / u.Mpc) * self.R).to('') ** 3)#* self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad) - (k * (1 / u.Mpc) * self.R).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f = (np.abs(f_2 * f_3) ** 2 * f_1).to('')
        return f

    def calc_sig(self):
        integrand = lambda k: self.func(k)* k**2 * self.getPk(k)
        f_1 = quad_vec(integrand, 1e-3, 100000, limit=100)
        for i in range(len(self.sig)):
            self.sig[i] = np.sqrt(f_1[0][i]).to('')
        return self.sig

    def func_der(self,k): #W(kR)*dW/dR* 1/2pi**2
        f_1 = 1/(2*np.pi**2) * u.Mpc # multiplying w/ Mpc to get units to cancel but they match analytically!
        f_2 =  3 / ((k * (1 / u.Mpc) * self.R).to('') ** 3)#
        f_3 = np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad) - (k * (1 / u.Mpc) * self.R).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R).to('') * u.rad) ##Window function
        f_4 = (-3 / self.R.value) * 1/u.Mpc *(u.Mpc / u.pc).to('') * np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f_5 = 3 * (k * (1 / u.Mpc)) * np.cos((k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f_6 = 1/u.Mpc * (k ** 2 * (1 / u.Mpc) * self.R).to('') * np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f = ((f_2 * np.abs(f_3)) * f_1 * f_2 * (f_4 + f_5 + f_6)).to('')
        return f #* self.phi(k)

    def der_sig(self): #Gives 2sig *dsig/dR
        integrand_der = lambda k: self.func_der(k) * k**2 * self.getPk(k)
        f_1 = quad_vec(integrand_der, 1e-3, 100000, limit=100)
        self.der = f_1[0]
        return self.der #units of 1/u.Mpc

    def find_Nl(self): #incorporating dR/dM in dlnsig/dM to get dn/dM
        for i in range(len(self.n_l)): ##n_l units of 1/Mpc**3
            self.den_n_l[i] = np.abs((4 / 3 * np.pi * (Rho_mean.to(u.M_sun/u.Mpc**3).value)) ** (-1 / 3) * 1/3 * (self.m_l[i]) ** (-2 / 3)* \
                              self.m_l[i]/(2*(self.sig[i]**2)) * self.der[i]) * \
                              np.sqrt(2/np.pi) * (Rho_mean.to(u.M_sun/u.Mpc**3).value)/self.m_l[i]*\
                              self.del_crit/self.sig[i]*np.exp(-self.del_crit**2/(2*self.sig[i]**2))

        #Calculating the normalization
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = np.insert(integr, 0, 0)
        N = np.diff(integr, prepend=0)
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        N = norm * N

        nlens = sum(N)
        ran_samp = np.random.choice(self.m_l, np.int64(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1

    def __post_init__(self):
        self.radius()
        self.calc_sig()
        self.der_sig()
        self.find_Nl()

@dataclass
class PressSchechter_test(MassFunction):
    Name: str = 'Press Schechter'
    m_l: list = field(default_factory=lambda: np.logspace(4, 12, 100))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    sig: list = field(default_factory=lambda: [1 for i in range(100)])
    der: list = field(default_factory=lambda: [1 for i in range(100)])
    R: list = field(default_factory=lambda: [1 for i in range(100)])
    del_crit: float = 1.686
    nparams: int = 1
    param_names: list = field(default_factory=lambda:['del_crit'])
    param_range: dict = field(default_factory=lambda:{'del_crit':(0.001,10)})

    def getPk(self,k): #Analytic fit from statsmodel to CLASS P(k)
        return 10**(1.5673598289074357 + np.log10(k)*-2.190072492256715 +np.log10(k)**2 *-0.46130280750744995)

    def radius(self):
        self.R = (3*self.m_l * u.M_sun/ (4 * np.pi * (Rho_mean.to(u.M_sun/u.pc**3)))) ** (1 / 3)
        return self.R

    def func(self,k, i):
        f_1 = k**2/(2*np.pi**2) #1/k  #In units of u.Mpc
        f_2 = 3 / ((k * (1 / u.Mpc) * self.R[i]).to('') ** 3)#* self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad) - (k * (1 / u.Mpc) * self.R[i]).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f = (np.abs(f_2 * f_3) ** 2 * f_1*self.getPk(k)).to('')
        return f

    def calc_sig(self):
        k = np.logspace(-3, 5, 100)
        int_val = []
        for i in range(len(self.R)):
            integrand = self.func(k, i)
            integr = scipy.integrate.trapezoid(integrand, k)
            int_val.append(integr)
        for i in range(len(self.sig)):
            self.sig[i] = np.sqrt(int_val[i])
        return self.sig

    def func_der(self,k,i): #W(kR)*dW/dR* 1/2pi**2
        f_1 = k**2/(2*np.pi**2) * u.Mpc # multiplying w/ Mpc to get units to cancel but they match analytically!
        f_2 =  3 / ((k * (1 / u.Mpc) * self.R[i]).to('') ** 3)#
        f_3 = np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad) - (k * (1 / u.Mpc) * self.R[i]).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R[i]).to('') * u.rad) ##Window function
        f_4 = (-3 / self.R[i].value) * 1/u.Mpc *(u.Mpc / u.pc).to('') * np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_5 = 3 * (k * (1 / u.Mpc)) * np.cos((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f_6 = 1/u.Mpc * (k ** 2 * (1 / u.Mpc) * self.R[i]).to('') * np.sin((k * (1 / u.Mpc) * self.R[i]).to('') * u.rad)
        f = ((f_2 * np.abs(f_3)) * f_1 * f_2 * (f_4 + f_5 + f_6) * self.getPk(k)).to('')
        return f #* self.phi(k)

    def der_sig(self): #Gives 2sig *dsig/dR
        k = np.logspace(-3, 5, 100)
        int_val = []
        for i in range(len(self.R)):
            integrand = self.func_der(k, i)
            integr = scipy.integrate.trapezoid(integrand, k)
            int_val.append(integr.value)
        #integr = scipy.integrate.cumulative_trapezoid(integrand, k, initial=integr[0])
        #der_val = np.diff(integr, prepend=integr[0])
        self.der = int_val
        return self.der #units of 1/u.Mpc

    def find_Nl(self): #incorporating dR/dM in dlnsig/dM to get dn/dM
        for i in range(len(self.n_l)): ##n_l units of 1/Mpc**3
            self.den_n_l[i] = np.abs((4 / 3 * np.pi * (Rho_mean.to(u.M_sun/u.Mpc**3).value)) ** (-1 / 3) * 1/3 * (self.m_l[i]) ** (-2 / 3)* \
                              self.m_l[i]/(2*(self.sig[i]**2)) * self.der[i]) * \
                              np.sqrt(2/np.pi) * (Rho_mean.to(u.M_sun/u.Mpc**3).value)/self.m_l[i]*\
                              self.del_crit/self.sig[i]*np.exp(-self.del_crit**2/(2*self.sig[i]**2))

        #Calculating the normalization
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = np.insert(integr, 0, 0)
        N = np.diff(integr, prepend=0)
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        N = norm * N

        nlens = sum(N)
        ran_samp = np.random.choice(self.m_l, np.int64(nlens), p=self.den_n_l / sum(self.den_n_l))
        c = Counter(ran_samp)
        nl = [c[m_l] for m_l in self.m_l]
        self.n_l = np.random.poisson(nl)
        if sum(self.n_l) == 0:
            self.n_l[random.randint(0,len(self.n_l)-1)] = 1

    def __post_init__(self):
        self.radius()
        self.calc_sig()
        self.der_sig()
        self.find_Nl()
