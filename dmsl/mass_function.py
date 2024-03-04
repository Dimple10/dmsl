'''
Defines mass functions
'''
import astropy.cosmology
from astropy.cosmology import WMAP7 as cosmo
import astropy.units as u
import numpy as np
import dmsl.galaxy_subhalo_dist as gsh
from dataclasses import dataclass,field
import dmsl.survey as surv
from scipy.integrate import quad_vec
from classy import Class
import scipy

RHO_CRIT = cosmo.critical_density(0.)
Rho_mean = cosmo.Om0 * RHO_CRIT
Rho_dm = gsh.density(8. * u.kpc)
h = cosmo.h

@dataclass #(example in survey.py)
class MassFunction:
    n_l : list = field(default_factory = list)
    m_l: list = field(default_factory = list)
    sur: surv = surv.Roman()

    @property
    def check_density(self):
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens**3 / 3.
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

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l, initial=integr[0])
        N = np.diff(integr, prepend=integr[0])
        N *= (u.Mpc) ** -3
        N = (N * vol).to('')
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = norm * N

    def __post_init__(self):
        self.find_Nl()


@dataclass
class Tinker(MassFunction):
    Name: str = 'Tinker'
    m_l: list = field(default_factory=lambda: np.logspace(4, 12, 20))
    den_n_l: list = field(default_factory=lambda: np.zeros((20)))
    n_l: list = field(default_factory=lambda: np.zeros((20)))
    nparams: int = 7
    A: float = 0.186  #FIXME Used values from colossus (https://bitbucket.org/bdiemer/colossus/src/master/colossus/lss/mass_function.py)
    a: float = 1.47
    b: float = 2.57
    c: float = 1.19
    sig: list = field(default_factory=lambda: [1 for i in range(20)])
    der: list = field(default_factory=lambda: [1 for i in range(20)])
    R: list = field(default_factory=lambda: [1 for i in range(20)])
    f: list = field(default_factory=lambda: [1 for i in range(20)])
    A_s: float = 2.105 * 10 ** -9
    n_s: float = 0.9665
    k_b: float = 13 * (1 / u.Mpc)  # Units of Mpc^-1
    n_b: float = 2.0  # or 3.0 (Fig #7 in Power of Halometry)
    k_s: float = 0.05 * (1 / u.Mpc)  # Units of Mpc^-1
    #cosmo:astropy.cosmology.Cosmology() = cosmo
    param_names: list = field(default_factory=lambda:['A', 'a', 'b', 'c', 'k_b', 'n_b', 'k_s'])
    param_range: dict = field(default_factory=lambda:{'A': (0, np.inf), 'a': (1, 100), 'b':(1, np.inf), 'c':(1, np.inf), 'k_b':(1,100),'n_b':(1,5), 'k_s':(0,1)})

    def getPk(self):
        cosmol = Class()
        cosmol.set({'P_k_max_1/Mpc':100,'omega_b':cosmo.Ob0*h**2,'omega_m':cosmo.Om0*h**2,'h':cosmo.h,'A_s':2.100549e-09,
                    'n_s':0.9660499,'tau_reio':0.05430842})
        cosmol.set({'output':'mPk'})
        cosmol.compute()
        kk = np.logspace(-3, np.log10(100), 10000)  # k in 1/Mpc
        Pk = [cosmol.pk(s,0.)*h**3 for s in kk]  # P(k) in (Mpc)**3

        F = scipy.interpolate.interp1d(np.log10(kk), np.log10(Pk), fill_value='extrapolate')
        return F

    def phi(self,k):
        return np.piecewise(k, [k<self.k_b*u.Mpc, k>=self.k_b*u.Mpc],
               [lambda k:self.A_s * (k / (u.Mpc * self.k_s)) ** (self.n_s - 1),
                lambda k:self.A_s * (self.k_b / self.k_s) ** (self.n_s - 1) * (k / (u.Mpc * self.k_b)) ** (self.n_b - 1)])

    def calc_f(self):
        self.f = (self.A * ((np.array(self.sig) / self.b) ** (-1 * self.a) + 1)) * np.exp(
            (-1 * self.c / np.array((self.sig)) ** 2))
       # print("f(sig)= ", self.f)

    def radius(self):
        self.R = (3*self.m_l * u.M_sun/ (4 * np.pi * (Rho_mean.to(u.M_sun/u.pc**3)))) ** (1 / 3)
        #print("R= ", self.R)
        return self.R

    def func(self,k):
        f_1 = k**2/(2*np.pi**2) #1/k  #In units of u.Mpc
        f_2 = 3 / ((k * (1 / u.Mpc) * self.R).to('') ** 3)#* self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad) - (k * (1 / u.Mpc) * self.R).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f = (np.abs(f_2 * f_3) ** 2 * f_1).to('')
        return f #* self.phi(k)

    def calc_sig(self):
        F = self.getPk()
        D2 = lambda k: 10**F(np.log10(k))*k**3/(2*np.pi**2*self.phi(k))
        integrand = lambda k: self.func(k)* 10**F(np.log10(k))#D2(k)**2 *self.phi(k)
        f_1 = quad_vec(integrand, 1e-3, 100000, limit=100)
        for i in range(len(self.sig)):
            self.sig[i] = np.sqrt(f_1[0][i]).to('')
        #self.sig = f_1[0]
        #print("Sig: ", self.sig)
        return self.sig

    def func_der(self,k):
        f_1 = k**2/(2*np.pi**2) * u.Mpc #1/k *u.Mpc #
        f_2 =  3 / ((k * (1 / u.Mpc) * self.R).to('') ** 3)# * self.D2[k] ** 2
        f_3 = np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad) - (k * (1 / u.Mpc) * self.R).to('') * np.cos(
            (k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f_4 = (-3 / self.R.value) * 1/u.Mpc *(u.Mpc / u.pc).to('') * np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f_5 = 3 * (k * (1 / u.Mpc)) * np.cos((k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f_6 = 1/u.Mpc * (k ** 2 * (1 / u.Mpc) * self.R).to('') * np.sin((k * (1 / u.Mpc) * self.R).to('') * u.rad)
        f = ((f_2 * np.abs(f_3)) * f_1 * f_2 * (f_4 + f_5 + f_6)).to('')
        return f #* self.phi(k)

    def der_sig(self):
        F = self.getPk()
        D2 = lambda k: 10 ** F(np.log10(k)) * k ** 3 / (2 * np.pi ** 2 * self.phi(k))
        integrand_der = lambda k: self.func_der(k) * 10**F(np.log10(k)) #D2(k)**2 *self.phi(k)
        f_1 = quad_vec(integrand_der, 1e-3, 100000, limit=100)
        self.der = f_1[0]
        #print("dsig/dR = ", self.der/self.sig)
        return self.der #units of 1/u.Mpc

    def find_Nl(self):
        for i in range(len(self.n_l)): ##n_l units of 1/Mpc**3
            self.den_n_l[i] = (4 / 3 * np.pi * (Rho_mean.to(u.M_sun/u.Mpc**3).value)) ** (-1 / 3) * 1/3 * (self.m_l[i]) ** (-2 / 3)* \
                              self.f[i] * (Rho_mean.to(u.M_sun/u.Mpc**3).value) / self.m_l[i] * (-self.der[i]/self.sig[i]**2) #*self.m_l[i]
        #Calculating the normalization
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l, initial=integr[0])
        N = np.diff(integr,prepend=integr[0])
        N *= (u.Mpc) ** -3
        N = (N * vol).to('')
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = norm * N

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
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 100))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    loga: float = np.log10(3.26*10**-5)
    b: float = -1.9
    logc: float =np.log10(2.57*10**7)
    nparams: int = 3
    param_names: list = field(default_factory=lambda:['loga', 'b', 'logc'])
    param_range: dict = field(default_factory=lambda: {'loga':(-9, 2), 'b': (-6, -0.1), 'logc':(1, 12)})

    def find_Nl(self):
        self.den_n_l = (10**self.loga) * ((self.m_l/(10**self.logc))** self.b) ##Units of M_sun^-1
        ##FIXME Higher by a factor of 10 than in the Shutz paper

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l,self.m_l)
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l,self.m_l,initial = integr[0])
        N = np.diff(integr, prepend=integr[0]) #CUMULATIVE TRAPZ ADDS PREVIOUS VAL TO EACH CURRENT VAL SO DIFF
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = norm * N

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
    m_wdm: float = 6.3 #in KeV
    omega_wdm: float = 0.25
    loga_cdm: float = np.log10(3.26*10**-5)
    b_cdm: float = -1.9
    logc_cdm: float = np.log10(2.57*10**7)
    a : float = 1.65*10**10
    b: float = -3.33
    c :float = 1.33
    d :float = 2.66
    nparams: int = 3 #m_wdm, gamma, beta
    param_names: list = field(default_factory=lambda:['m_wdm', 'gamma', 'beta'])#, 'loga_cdm','b_cdm','logc_cdm'])
    param_range: dict = field(default_factory=lambda:{'m_wdm': (0.01,200),'gamma': (0.01, 50), 'beta':(0,3)})#,
                                                     # 'loga_cdm':(-8, 2), 'b_cdm': (-4, -0.01), 'logc_cdm':(4, 10)})
    def calc_Mhm(self):
        self.M_hm = self.a * (self.m_wdm)** self.b * (self.omega_wdm/0.25)**self.c #* (h/0.7)**2.66

    def find_Nl(self):
        cdm = CDM(m_l=self.m_l)#, loga=self.loga_cdm, b=self.b_cdm, logc=self.logc_cdm) #return dN/dM, we need dN/dlnM
        cdm_den_nl = cdm.den_n_l * cdm.m_l
        self.den_n_l = (1 + self.gamma*self.M_hm/self.m_l)**(-1*self.beta) * cdm_den_nl
        #Convert from dN/dlnM to dN/dM--
        self.den_n_l = self.den_n_l / self.m_l

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l, initial=integr[0])
        N = np.diff(integr, prepend=integr[0])
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = norm * N

    def __post_init__(self):
        self.calc_Mhm()
        self.find_Nl()

@dataclass
class WDM_lensing(MassFunction):
    Name: str = 'Warm Dark Matter Lensing'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 100))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    M_hm: float = 1.0
    beta: float = 1.3 #Shutz reference
    m_wdm: float = 6.3 #in KeV
    omega_wdm: float = cosmo.Odm0
    a: float = 1.65 * 10 ** 10
    b: float = -3.33
    c: float = 1.33
    d: float = 2.66
    param_names: list = field(default_factory=lambda:['alpha', 'M_0']) #FIXME To correct vals if needed
    param_range: dict = field(default_factory=lambda:{'alpha': (0.1, 2), 'M_0': (1, 100)})

    def calc_Mhm(self):
        self.M_hm = self.a * self.m_wdm**self.b * (self.omega_wdm/0.25)**self.c * (h/0.7)**self.d
        #print(self.M_hm)

    def find_Nl(self):
        cdm = CDM(m_l=self.m_l) #return dN/dM, we need dN/dlnM
        cdm_den_nl = cdm.den_n_l * cdm.m_l
        self.den_n_l = (1 + self.M_hm/self.m_l)**(-1*self.beta) * cdm_den_nl
        #Convert from dN/dlnM to dN/dM--
        self.den_n_l = self.den_n_l / self.m_l

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l)
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l, initial=integr[0])
        N = np.diff(integr, prepend=integr[0])
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        print(norm, m_dm)
        self.n_l = norm * N

    def __post_init__(self):
        self.calc_Mhm()
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
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l, initial=integr[0])
        N = np.diff(integr,prepend=integr[0])
        N *= (u.Mpc) ** -3
        N = (N * vol).to('')
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = norm * N

    def __post_init__(self):
        self.radius()
        self.calc_sig()
        self.der_sig()
        self.find_Nl()

@dataclass
class PressSchechter_test(MassFunction):
    Name: str = 'Press Schechter'
    m_l: list = field(default_factory=lambda: np.logspace(4, 12, 10))
    den_n_l: list = field(default_factory=lambda: np.zeros((10)))
    n_l: list = field(default_factory=lambda: np.zeros((10)))
    sig: list = field(default_factory=lambda: [1 for i in range(10)])
    der: list = field(default_factory=lambda: [1 for i in range(10)])
    R: list = field(default_factory=lambda: [1 for i in range(10)])
    del_crit: float = 1.686
    nparams: int = 1
    param_names: list = field(default_factory=lambda:['del_crit'])
    param_range: dict = field(default_factory=lambda:{'del_crit':(0.01,4)})

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
        k = np.logspace(-3, 5, 10)
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
        k = np.logspace(-3, 5, 10)
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
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l, self.m_l, initial=integr[0])
        N = np.diff(integr,prepend=integr[0])
        N *= (u.Mpc) ** -3
        N = (N * vol).to('')
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = norm * N

    def __post_init__(self):
        self.radius()
        self.calc_sig()
        self.der_sig()
        self.find_Nl()
