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
    name: str ='Test'
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
        if( not self.check_density()):
            raise RuntimeWarning("Check density failed! Retry with different mass values")
        else:
            pass

    def update(self):
        self.__post_init__()


@dataclass
class PowerLaw(MassFunction):
    Name: str = 'Power Law'
    m_l : list = field(default_factory=lambda:np.logspace(0,2,10))
    n_l: list = field(default_factory=lambda: np.zeros((10)))  ##FIXME len(m_l)
    alpha: int = 1
    M_0: float = 1
    nparams: int = 2

    def find_Nl(self):
        #Calculating the normalized number of lenses
        for i in range(len(self.m_l)):
            self.n_l[i] = np.ceil((self.m_l[i] / self.M_0) ** self.alpha)

        ## Calculating the normalization to fit Total DM Density
        sum = np.sum(self.m_l*self.n_l)
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        m_dm = (Rho_dm * vol).to(u.Msun)
        A = (m_dm / sum)
        A = A.to_value()
        self.n_l = np.ceil(A*self.n_l)
        return self.n_l

    def __post_init__(self):
        self.find_Nl()
        pass


@dataclass
class test_Tinker(MassFunction):
    Name: str = 'Tinker'
    m_l: list = field(default_factory=lambda: np.logspace(4, 12, 100))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    nparams: int = 7
    A: float = 0.186  ###FIXME Used values from colossus (https://bitbucket.org/bdiemer/colossus/src/master/colossus/lss/mass_function.py)
    a: float = 1.47
    b: float = 2.57
    c: float = 1.19
    sig: list = field(default_factory=lambda: [1 for i in range(100)])
    #k: list = field(default_factory=lambda: np.logspace(-3,np.log10(100000), 1000))
    der: list = field(default_factory=lambda: [1 for i in range(100)])
    R: list = field(default_factory=lambda: [1 for i in range(100)])
    f: list = field(default_factory=lambda: [1 for i in range(100)])
    A_s: float = 2.105 * 10 ** -9
    n_s: float = 0.9665
    k_b: float = 13 * (1 / u.Mpc)  # Units of Mpc^-1
    n_b: float = 2.0  # or 3.0 (Fig #7 in Power of Halometry)
    k_s: float = 0.05 * (1 / u.Mpc)  # Units of Mpc^-1
    #cosmo:astropy.cosmology.Cosmology() = cosmo

    def getPk(self):
        cosmol = Class()
        cosmol.set({'P_k_max_1/Mpc':100,'omega_b':cosmo.Ob0*h**2,'omega_m':cosmo.Om0*h**2,'h':cosmo.h,'A_s':2.100549e-09,
                    'n_s':0.9660499,'tau_reio':0.05430842})
        cosmol.set({'output':'mPk'})
        cosmol.compute()
        kk = np.logspace(-3, np.log10(100), 10000)  # k in 1/Mpc
        Pk = [cosmol.pk(s,0.)*h**3 for s in kk]  # P(k) in (Mpc)**3
        #for s in kk:
        #    Pk.append(cosmol.pk(s, 0.) * h** 3 ) # function .pk(k,z)

        F = scipy.interpolate.interp1d(np.log10(kk), np.log10(Pk), fill_value='extrapolate')
        return F
        #self.D2 = 10**F(np.log10(self.k))*self.k**3/(2*np.pi**2*self.phi(self.k)) #Phi with dimensions
        #print(D2)

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

'''
@dataclass
class Tinker(MassFunction):
    Name: str = 'Tinker'
    m_l : list = field(default_factory=lambda:np.logspace(0,2,100))
    a: float = 1.47
    b: float = 2.57
    sig: list = field(default_factory=lambda:[1 for i in range(100)])
    der: list = field(default_factory=lambda:[1 for i in range(100)])
    A: float = 0.186 ###FIXME Used values from colossus (https://bitbucket.org/bdiemer/colossus/src/master/colossus/lss/mass_function.py)
    c: float = 1.19
    R : list = field(default_factory=lambda:[1 for i in range(100)])
    n_l :list = field(default_factory=lambda: np.zeros((100)))
    f : list = field(default_factory=lambda:[1 for i in range(100)])
    A_s: float = 2.105 * 10 ** -9
    n_s: float = 0.9665
    k_b: float = 13*(1/u.Mpc)  # Units of Mpc^-1
    n_b: float = 2.0  # or 3.0 (Fig #7 in Power of Halometry)
    k_s: float = 0.05*(1/u.Mpc)  # Units of Mpc^-1
    D: float = 10**10

    def radius(self):
        self.R = (3*self.m_l*u.M_sun/(4*np.pi*(Rho_mean.to(u.M_sun/u.pc**3))))**(1/3) ##Rho_mean in g/cm**
        return self.R

    def func_high(self,k):
        pk = 2*np.pi**2 * 1/k**3* (self.A_s * (self.k_b/self.k_s)**(self.n_s-1) * (k/(u.Mpc*self.k_b))**(self.n_b-1))
        pk_d = (self.A_s * (self.k_b/self.k_s)**(self.n_s-1) * (k/(u.Mpc*self.k_b))**(self.n_b-1))
        f_2 = 1/k * self.D**2 * 3/(k *(1/u.Mpc)*self.R).to('') **3
        f_3 = np.sin((k*(1/u.Mpc)*self.R).to('')*u.rad) - (k*(1/u.Mpc)*self.R) * np.cos((k*(1/u.Mpc)*self.R).to('')*u.rad)
        f = (pk_d * np.abs(f_2 * f_3)**2 ).to('')
        return f
    
    def func_low(self,k):
        pk = 2*np.pi**2 * 1/k**3 * (self.A_s * (k/(u.Mpc*self.k_s))**(self.n_s-1))
        pk_d = (self.A_s * (k/(u.Mpc*self.k_s))**(self.n_s-1))
        f_2 = 1/k * self.D**2 * 3/((k *(1/u.Mpc)*self.R).to('') **3)
        f_3 = np.sin((k*(1/u.Mpc)*self.R).to('')*u.rad) - (k*(1/u.Mpc)*self.R) * np.cos((k*(1/u.Mpc)*self.R).to('')*u.rad)
        f = (pk_d * np.abs(f_2 * f_3)**2).to('')
        return f

    def calc_sig(self):
        f_1 = quad_vec(self.func_low, 1e-3 , self.k_b.value, limit = 100)
        f_2 = quad_vec(self.func_high, self.k_b.value, 100000, limit = 100)
        for i in range(len(self.sig)):
            self.sig[i] = np.sqrt(f_1[0][i] + f_2[0][i])
        #print("Sig: ", self.sig)
        return self.sig

    def calc_f(self):
        self.f = (self.A*((np.array(self.sig)/self.b)**(-1*self.a) + 1)) * np.exp((-1*self.c/np.array((self.sig))**2))

    def func_low_der(self,k):
        f_1 = (1 / k * u.Mpc) * (self.A_s * (k / (self.k_s*(u.Mpc))) ** (self.n_s - 1)) * self.D ** 2
        f_2 = 3 / ((k *(1/u.Mpc)* self.R).to('') ** 3)
        f_3 = np.sin((k * (1/u.Mpc)*self.R).to('')*u.rad) - (k *(1/u.Mpc)* self.R) * np.cos((k * (1/u.Mpc)*self.R).to('')*u.rad)
        f_4 = (-3/self.R.value) *(1/u.pc)* np.sin((k * (1/u.Mpc)*self.R).to('')*u.rad)
        f_5 = 3 * (k*(1/u.Mpc)) * np.cos((k * (1/u.Mpc)*self.R).to('')*u.rad)
        f_6 = (k**2 * (1/u.Mpc)**2 * self.R) * np.cos((k * (1/u.Mpc)*self.R).to('')*u.rad)

        f = (f_1 * (f_2 * f_3)**2 * f_2 * (f_4 + f_5 + f_6)).to('')
        return f

    def func_high_der(self, k):
        f_1 = (1 / k * u.Mpc) * (self.A_s * (self.k_b / self.k_s) ** (self.n_s - 1) * (k / (u.Mpc*self.k_b) ** (self.n_b - 1))) * self.D ** 2
        f_2 = 3 / ((k * (1/u.Mpc)*self.R).to('') ** 3)
        f_3 = np.sin((k * (1/u.Mpc)* self.R).to('')*u.rad) - (k * (1/u.Mpc)* self.R) * np.cos((k * (1/u.pc)* self.R).to('')*u.rad)
        f_4 = (-3/self.R.value) * (1/u.pc) * np.sin((k * (1/u.Mpc)*self.R).to('')*u.rad)
        f_5 = 3 * (k*(1/u.Mpc)) * np.cos((k * (1/u.Mpc)*self.R).to('')*u.rad)
        f_6 = (k**2 * (1/u.Mpc)**2 * self.R) * np.cos((k * (1/u.Mpc)*self.R).to('')*u.rad)

        f = (f_1 * (f_2 * f_3)**2 * f_2 * (f_4 + f_5 + f_6)).to('')
        return f

    def der_sig(self):
        f_1= quad_vec(self.func_low_der, 1e-3, self.k_b.value, limit = 100)
        f_2 = quad_vec(self.func_high_der, self.k_b.value, 100000, limit = 100)
        self.der = (f_1[0]+ f_2[0])
        #print("Der: ", self.der)
        return self.der

    def find_Nl(self):
        for i in range(len(self.n_l)):
            self.n_l[i] = np.ceil(self.f[i]* (Rho_mean.value)/self.m_l[i] * 1/3*(4*np.pi*(Rho_mean.value))**(1/3) * (self.m_l[i])**(-4/3) * self.der[i])
        ## Calculating the normalization to fit Total DM Density
        sum = np.sum(self.m_l * self.n_l)
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        m_dm = (Rho_dm * vol).to(u.Msun)

        return self.n_l

    def __post_init__(self):
        self.radius()
        self.calc_sig()
        self.calc_f()
        self.der_sig()
        self.find_Nl()
        pass
'''

@dataclass
class CDM(MassFunction):
    Name: str = 'CDM'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 100))
    den_n_l: list = field(default_factory=lambda: np.zeros((100)))
    n_l: list = field(default_factory=lambda: np.zeros((100)))
    a: float = 3.26*10**-5
    b: float = -1.9
    c: float = 2.57*10**7
    nparams: int = 3

    def find_Nl(self):
        self.den_n_l = self.a * (self.m_l/(self.c))** self.b ##Units of M_sun^-1
        ##FIXME Higher by a factor of 10 than in the Shutz paper

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l,self.m_l)
        integr = scipy.integrate.cumulative_trapezoid(self.den_n_l,self.m_l,initial = integr[0])
        N = np.diff(integr, prepend=integr[0])
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        print("m-sur: ", m_sur)
        print('m_dm: ', m_dm)
        norm = m_sur / m_dm
        print('Norm:', norm)
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
    a : float = 1.65*10**10
    b: float = -3.33
    c :float = 1.33
    d :float = 2.66
    nparams: int = 3 #m_wdm, gamma, beta

    def calc_Mhm(self):
        self.M_hm = self.a * self.m_wdm** self.b * (self.omega_wdm/0.25)**self.c #* (h/0.7)**2.66

    def find_Nl(self):
        cdm = CDM(m_l=self.m_l) #return dN/dM, we need dN/dlnM
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

    def calc_Mhm(self):
        self.M_hm = self.a * self.m_wdm**self.b * (self.omega_wdm/0.25)**self.c * (h/0.7)**self.d
        print(self.M_hm)

    def find_Nl(self):
        cdm = CDM(m_l=self.m_l) #return dN/dM, we need dN/dlnM
        cdm_den_nl = cdm.den_n_l * cdm.m_l
        self.den_n_l = (1 + self.M_hm/self.m_l)**(-1*self.beta) * cdm_den_nl
        #Convert from dN/dlnM to dN/dM--
        self.den_n_l = self.den_n_l / self.m_l

        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        new_n = []
        new_m = 10 ** ((np.log10(self.m_l[1:]) + np.log10(self.m_l[:-1])) / 2)
        for i in range(len(self.den_n_l) - 1):
            mtemp = [self.m_l[i], self.m_l[i + 1]]
            ntemp = [self.den_n_l[i], self.den_n_l[i + 1]]
            n = scipy.interpolate.interp1d(np.log10(mtemp), np.log10(ntemp))
            integrand = lambda m: 10 ** n(np.log10(m))
            new_n.append(scipy.integrate.quad(integrand, mtemp[0], mtemp[1])[0])
        temp = scipy.interpolate.interp1d(np.log10(new_m), np.log10(new_n), fill_value='extrapolate')
        N = 10 ** temp(np.log10(self.m_l))
        m_dm = np.sum(N * self.m_l) * u.Msun
        m_sur = Rho_dm * vol
        norm = m_sur / m_dm
        self.n_l = norm * N

    def __post_init__(self):
        self.calc_Mhm()
        self.find_Nl()

@dataclass
class PressSchehter(MassFunction):
    Name: str = 'Press Schechter'
    m_l: list = field(default_factory=lambda: np.logspace(0, 2, 10))
    n_l: list = field(default_factory=lambda: np.zeros((10)))
    delta: float = 1
    D: float = 1
    sig: list = field(default_factory=lambda: [1 for i in range(len(m_l))])
    der: list = field(default_factory=lambda: [1 for i in range(len(m_l))])

    def calc_delta(self):
        return self.delta/self.D

    def calc_sig(self): ##FIXME LEFT SAME AS TINKER FOR NOW
        func_low = 1/k * (self.A_s * (k/self.k_s)**(self.n_s-1)) * self.D**2 * np.abs(3/(k*self.R)**3 * (np.sin(k*self.R)-k*self.R*np.cos(k*self.R)))**2
        func_high = 1/k * (self.A_s * (self.k_b/self.k_s)**(self.n_s-1) * (k/self.k_b)**(self.n_b-1)) * self.D**2 * np.abs(3/(k*self.R)**3 * (np.sin(k*self.R)-k*self.R*np.cos(k*self.R)))**2
        self.sig = np.sqrt(quad(func_low, 1e-3, k_b) + quad (func_high, k_b, 1000))
        return self.sig

    def der_sig(self): ##FIXME LEFT SAME AS TINKER FOR NOW, needs M/sig * dSig/dM
        func_low = 1 / k * (self.A_s * (k / self.k_s) ** (self.n_s - 1)) * self.D ** 2 * (3 / (k * self.R) ** 3 * (np.sin(k * self.R) - k * self.R * np.cos(k * self.R))) ** 2 * (3/(k*self.R)** 3) * ((-3/self.R)*np.sin(k*self.R) + 3*k*np.cos(k*self.R) + k**2 * self.R * np.cos(k*self.R))
        func_high = 1 / k * (self.A_s * (self.k_b / self.k_s) ** (self.n_s - 1) * (k / self.k_b) ** (self.n_b - 1)) * self.D ** 2 * (3 / (k * self.R) ** 3 * (np.sin(k * self.R) - k * self.R * np.cos(k * self.R))) ** 2 * (3/(k*self.R)** 3) * ((-3/self.R)*np.sin(k*self.R) + 3*k*np.cos(k*self.R) + k**2 * self.R * np.cos(k*self.R))
        self.der = np.sqrt(quad(func_low, 1e-3, k_b) + quad(func_high, k_b, 1000))
        return self.der

    def find_Nl(self):
        for i in range(len(self.n_l)):
            self.n_l[i] = np.ceil(np.sqrt(2/np.pi) * (Rho_mean/self.m_l[i]) * self.delta/self.sig[i] * np.exp(-(self.delta**2/(self.sig[i])**2)) * np.abs(self.der[i]))

        ## Calculating the normalization to fit Total DM Density
        sum = np.sum(self.m_l * self.n_l)
        vol = self.sur.fov_rad ** 2 * self.sur.maxdlens ** 3 / 3.
        m_dm = (Rho_dm * vol).to(u.Msun)
        A = (m_dm / sum)
        A = A.to_value()
        self.n_l *= A
        return self.n_l

    def __post_init__(self):
        self.calc_delta()
        self.calc_sig()
        self.der_sig()
        self.find_Nl()

