import numpy as np
import scipy.stats as stats
import scipy.integrate as si
import scipy.optimize as so
import scipy.interpolate as interp
import scipy.linalg as sla
import pandas as pd
import itertools
import sys

class BivariateClayton:

    """
    Type of bivariate copula that exhibits strong left-tail dependence
    """

    def __init__(self,theta):
        """
        param: theta: dependence parameter
        """
        self.theta = theta
        self.support = {'theta':[0,50]} # (0,inf) in reality, but overflows after 50

    def pdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability density at point (u,v)
        """
        return (1+self.theta)*(u*v)**(-1-self.theta)*(u**(-self.theta)+v**(-self.theta)-1)**(-1/self.theta - 2)

    def cdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that U ≤ u and V ≤ v
        """
        return (u**(-self.theta) + v**(-self.theta) - 1)**(-1/self.theta)

    def cdf_v_given_u(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that V ≤ v given U = u
        """
        return u**(-self.theta-1)*((v**(-self.theta)+u**(-self.theta)-1)**(-1-1/self.theta))

    def inv_cdf_v_given_u(self,u,p):
        """
        param: u: uniform random variable on [0,1]
        param: p: probability that V ≤ v given U = u
        returns: v: value of v such that Pr(V ≤ v | U = u) = p
        """
        return((u**(-self.theta)*(p**(-self.theta/(1 + self.theta)) - 1) + 1)**(-1/self.theta))

    def draw_v_given_u(self,u):
        """
        param: u: specified values of u
        """
        p = stats.uniform(loc=0,scale=1).rvs(size=u.shape)
        v = self.inv_cdf_v_given_u(u,p)
        return(v)

    def draw_uv_pairs(self,n):
        """
        param: n: number of (u,v) pairs to draw
        """
        u = stats.uniform(loc=0,scale=1).rvs(size=n)
        v = self.draw_v_given_u(u)
        return(u,v)

    def kendalls_tau(self):
        """
        returns: tau_k: kendall rank correlation coefficient
        """
        tau_k = self.theta/(self.theta + 2)
        return(tau_k)

    def spearmans_rho(self):
        """
        returns: rho_s: spearman rank correlation coefficient
        """
        f = lambda u,v: self.cdf(u,v) - u*v
        y,err = si.dblquad(f,0,1,0,1,epsabs=1e-4,epsrel=1e-4)
        rho_s = 12*y
        return(rho_s)

class BivariateFrank:

    """
    Type of bivariate copula that exhibits weak tail-dependence
    """

    def __init__(self,theta):
        """
        param: theta: dependence parameter
        """
        self.theta = theta
        self.support = {'theta':[0,25]} # (0,inf) in reality, but overflows after 300

    def pdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability density at point (u,v)
        """
        return self.theta*(np.exp(self.theta) - 1)*np.exp(self.theta*(u + v) + self.theta)/(-(np.exp(self.theta) - 1)*np.exp(self.theta*(u + v)) + (np.exp(self.theta*u) - 1)*(np.exp(self.theta*v) - 1)*np.exp(self.theta))**2

    def cdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that U ≤ u and V ≤ v
        """
        return -np.log(1 + (-1 + np.exp(-self.theta*u))*(-1 + np.exp(-self.theta*v))/(-1 + np.exp(-self.theta)))/self.theta

    def cdf_v_given_u(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that V ≤ v given U = u
        """
        return (-1 + np.exp(-self.theta*v))*np.exp(-self.theta*u)/((-1 + np.exp(-self.theta))*(1 + (-1 + np.exp(-self.theta*u))*(-1 + np.exp(-self.theta*v))/(-1 + np.exp(-self.theta))))

    def inv_cdf_v_given_u(self,u,p):
        """
        param: u: uniform random variable on [0,1]
        param: p: probability that V ≤ v given U = u
        returns: v: value of v such that Pr(V ≤ v | U = u) = p
        """
        return np.log((p*np.exp(self.theta*u) - p + 1)*np.exp(self.theta)/(-p*np.exp(self.theta) + p*np.exp(self.theta*u) + np.exp(self.theta)))/self.theta

    def draw_v_given_u(self,u):
        """
        param: u: specified values of u
        """
        p = stats.uniform(loc=0,scale=1).rvs(size=u.shape)
        v = self.inv_cdf_v_given_u(u,p)
        return(v)

    def draw_uv_pairs(self,n):
        """
        param: n: number of (u,v) pairs to draw
        """
        u = stats.uniform(loc=0,scale=1).rvs(size=n)
        v = self.draw_v_given_u(u)
        return(u,v)

    def kendalls_tau(self):
        """
        returns: tau_k: kendall rank correlation coefficient
        """
        f = lambda u,v: self.cdf_v_given_u(u,v)*self.cdf_v_given_u(v,u)
        y,err = si.dblquad(f,0,1,0,1,epsabs=1e-4,epsrel=1e-4)
        tau_k = 1-4*y
        return(tau_k)

    def spearmans_rho(self):
        """
        returns: rho_s: spearman rank correlation coefficient
        """
        f = lambda u,v: self.cdf(u,v) - u*v
        y,err = si.dblquad(f,0,1,0,1,epsabs=1e-4,epsrel=1e-4)
        rho_s = 12*y
        return(rho_s)

class BivariateGumbel:

    """
    Type of bivariate copula that exhibits strong right-tail dependence
    """

    def __init__(self,theta):
        """
        param: theta: dependence parameter
        """
        self.theta = theta
        self.support = {'theta':[1,50]} # [1,inf) in reality, but overflows after 50

    def pdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability density at point (u,v)
        """
        return (-np.log(u))**self.theta*(-np.log(v))**self.theta*((-np.log(u))**self.theta + (-np.log(v))**self.theta)**(1/self.theta)*(self.theta + ((-np.log(u))**self.theta + (-np.log(v))**self.theta)**(1/self.theta) - 1)*np.exp(-((-np.log(u))**self.theta + (-np.log(v))**self.theta)**(1/self.theta))/(u*v*((-np.log(u))**(2*self.theta) + 2*(-np.log(u))**self.theta*(-np.log(v))**self.theta + (-np.log(v))**(2*self.theta))*np.log(u)*np.log(v))

    def cdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that U ≤ u and V ≤ v
        """
        return np.exp(-((-np.log(u))**self.theta + (-np.log(v))**self.theta)**(1/self.theta))

    def cdf_v_given_u(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that V ≤ v given U = u
        """
        return -(-np.log(u))**self.theta*((-np.log(u))**self.theta + (-np.log(v))**self.theta)**(-1 + 1/self.theta)*np.exp(-((-np.log(u))**self.theta + (-np.log(v))**self.theta)**(1/self.theta))/(u*np.log(u))

    def inv_cdf_v_given_u(self,u,p):
        """
        param: u: uniform random variable on [0,1]
        param: p: probability that V ≤ v given U = u
        returns: v: value of v such that Pr(V ≤ v | U = u) = p
        """
        xtol=1e-4
        rtol=1e-4
        smallnum = np.finfo(float).eps
        a=smallnum
        b=1.0 - smallnum
        f = lambda v,u,p: self.cdf_v_given_u(u,v) - p
        v = np.array([so.bisect(f,a=a,b=b,xtol=xtol,rtol=rtol,args=(u[i],p[i])) for i in range(len(u))])
        return v

    def draw_v_given_u(self,u):
        """
        param: u: specified values of u
        """
        p = stats.uniform(loc=0,scale=1).rvs(size=u.shape)
        v = self.inv_cdf_v_given_u(u,p)
        return(v)

    def draw_uv_pairs(self,n):
        """
        param: n: number of (u,v) pairs to draw
        """
        u = stats.uniform(loc=0,scale=1).rvs(size=n)
        v = self.draw_v_given_u(u)
        return(u,v)

    def kendalls_tau(self):
        """
        returns: tau_k: kendall rank correlation coefficient
        """
        tau_k = (self.theta - 1)/self.theta
        return(tau_k)

    def spearmans_rho(self):
        """
        returns: rho_s: spearman rank correlation coefficient
        """
        f = lambda u,v: self.cdf(u,v) - u*v
        y,err = si.dblquad(f,0,1,0,1,epsabs=1e-4,epsrel=1e-4)
        rho_s = 12*y
        return(rho_s)

class BivariateJoe:

    """
    Type of bivariate copula that exhibits strong right-tail dependence
    """

    def __init__(self,theta):
        """
        param: theta: dependence parameter
        """
        self.theta = theta
        self.support = {'theta':[1,50]} # [1,inf) in reality, but overflows after 50

    def pdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability density at point (u,v)
        """
        return (1 - u)**(self.theta - 1)*(1 - v)**(self.theta + 1)*(self.theta*(-(1 - u)**self.theta*(1 - v)**self.theta + (1 - u)**self.theta + (1 - v)**self.theta)**(1/self.theta) + (self.theta - 1)*((1 - u)**self.theta - 1)*((1 - v)**self.theta - 1)*(-(1 - u)**self.theta*(1 - v)**self.theta + (1 - u)**self.theta + (1 - v)**self.theta)**((1 - self.theta)/self.theta))/((v - 1)**2*(-(1 - u)**self.theta*(1 - v)**self.theta + (1 - u)**self.theta + (1 - v)**self.theta))

    def cdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that U ≤ u and V ≤ v
        """
        return 1 - (-(1 - u)**self.theta*(1 - v)**self.theta + (1 - u)**self.theta + (1 - v)**self.theta)**(1/self.theta)

    def cdf_v_given_u(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that V ≤ v given U = u
        """
        return (1 - u)**(self.theta - 1)*(1 - (1 - v)**self.theta)*(-(1 - u)**self.theta*(1 - v)**self.theta + (1 - u)**self.theta + (1 - v)**self.theta)**((1 - self.theta)/self.theta)

    def inv_cdf_v_given_u(self,u,p):
        """
        param: u: uniform random variable on [0,1]
        param: p: probability that V ≤ v given U = u
        returns: v: value of v such that Pr(V ≤ v | U = u) = p
        """
        xtol=1e-4
        rtol=1e-4
        smallnum = np.finfo(float).eps
        a=smallnum
        b=1.0 - smallnum
        f = lambda v,u,p: self.cdf_v_given_u(u,v) - p
        v = np.array([so.bisect(f,a=a,b=b,xtol=xtol,rtol=rtol,args=(u[i],p[i])) for i in range(len(u))])
        return v

    def draw_v_given_u(self,u):
        """
        param: u: specified values of u
        """
        p = stats.uniform(loc=0,scale=1).rvs(size=u.shape)
        v = self.inv_cdf_v_given_u(u,p)
        return(v)

    def draw_uv_pairs(self,n):
        """
        param: n: number of (u,v) pairs to draw
        """
        u = stats.uniform(loc=0,scale=1).rvs(size=n)
        v = self.draw_v_given_u(u)
        return(u,v)

    def kendalls_tau(self):
        """
        returns: tau_k: kendall rank correlation coefficient
        """
        f = lambda u,v: self.cdf_v_given_u(u,v)*self.cdf_v_given_u(v,u)
        y,err = si.dblquad(f,0,1,0,1,epsabs=1e-4,epsrel=1e-4)
        tau_k = 1-4*y
        return(tau_k)

    def spearmans_rho(self):
        """
        returns: rho_s: spearman rank correlation coefficient
        """
        f = lambda u,v: self.cdf(u,v) - u*v
        y,err = si.dblquad(f,0,1,0,1,epsabs=1e-4,epsrel=1e-4)
        rho_s = 12*y
        return(rho_s)

class BivariateGaussian:

    """
    Type of elliptic copula that exhibits weak tail-dependence
    """

    def __init__(self,theta):
        """
        param: theta: dependence parameter
        """
        self.theta = theta
        self.support = {'theta':[-1,1]}

        # Extra attributes specific to gaussian distribution
        self.multivariate_normal_pdf = np.vectorize(lambda x,y: stats.multivariate_normal(cov=np.array([[1,self.theta],[self.theta,1]])).pdf([x,y]))
        self.multivariate_normal_cdf = np.vectorize(lambda x,y: stats.multivariate_normal(cov=np.array([[1,self.theta],[self.theta,1]])).cdf([x,y]))

    def pdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability density at point (u,v)
        """
        d = stats.norm()
        x = d.ppf(u)
        y = d.ppf(v)
        smallnum = np.finfo(float).eps
        numerator = self.multivariate_normal_pdf(x,y)
        denominator = np.exp(d.logpdf(x) + d.logpdf(y))
        return np.maximum(numerator/(denominator+smallnum),smallnum)

    def cdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that U ≤ u and V ≤ v
        """
        d = stats.norm()
        return self.multivariate_normal_cdf(d.ppf(u),d.ppf(v))

    def cdf_v_given_u(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that V ≤ v given U = u
        """
        d = stats.norm()
        z = (d.ppf(v) - self.theta*d.ppf(u))/np.sqrt(1 - self.theta**2)
        return d.cdf(z)

    def inv_cdf_v_given_u(self,u,p):
        """
        param: u: uniform random variable on [0,1]
        param: p: probability that V ≤ v given U = u
        returns: v: value of v such that Pr(V ≤ v | U = u) = p
        """
        d = stats.norm()
        z = d.ppf(p)*np.sqrt(1-self.theta**2)+self.theta*d.ppf(u)
        return d.cdf(z)

    def draw_v_given_u(self,u):
        """
        param: u: specified values of u
        """
        p = stats.uniform(loc=0,scale=1).rvs(size=u.shape)
        v = self.inv_cdf_v_given_u(u,p)
        return(v)

    def draw_uv_pairs(self,n):
        """
        param: n: number of (u,v) pairs to draw
        """
        u = stats.uniform(loc=0,scale=1).rvs(size=n)
        v = self.draw_v_given_u(u)
        return(u,v)

    def kendalls_tau(self):
        """
        returns: tau_k: kendall rank correlation coefficient
        """
        tau_k = 2/np.pi*np.arcsin(self.theta)
        return(tau_k)

    def spearmans_rho(self):
        """
        returns: rho_s: spearman rank correlation coefficient
        """
        rho_s = 6/np.pi*np.arcsin(self.theta/2)
        return(rho_s)

class BivariateStudentsT:

    """
    Type of elliptic copula that exhibits strong tail-dependence
    """

    def __init__(self,theta,df):
        """
        param: theta: dependence parameter
        param: df: degrees of freedom
        """
        self.theta = theta
        self.df = df
        self.support = {'theta':[-1,1],'df':[0,100]}

        # Extra attributes specific to t distribution
        self.multivariate_t_pdf = np.vectorize(lambda x,y: stats.multivariate_t(shape=np.array([[1,self.theta],[self.theta,1]]),df=df).pdf([x,y]))
        self.multivariate_t_cdf = np.vectorize(lambda x,y: si.quad(lambda s: stats.chi2(df=self.df).pdf(s)*stats.multivariate_normal(cov=np.array([[1,self.theta],[self.theta,1]])).cdf([x*np.sqrt(s/self.df),y*np.sqrt(s/self.df)]),0,np.inf,epsabs=1e-3,epsrel=1e-3)[0])

    def pdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability density at point (u,v)
        """
        d = stats.t(df=self.df)
        x = d.ppf(u)
        y = d.ppf(v)
        smallnum = np.finfo(float).eps
        numerator = self.multivariate_t_pdf(x,y)
        denominator = np.exp(d.logpdf(x) + d.logpdf(y))
        return np.maximum(numerator/(denominator+smallnum),smallnum)

    def cdf(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that U ≤ u and V ≤ v
        """
        d = stats.t(df=self.df)
        return self.multivariate_t_cdf(d.ppf(u),d.ppf(v))

    def cdf_v_given_u(self,u,v):
        """
        param: u: uniform random variable on [0,1]
        param: v: uniform random variable on [0,1]
        returns: probability that V ≤ v given U = u
        """
        d1 = stats.t(df=self.df)
        x1 = d1.ppf(u)

        # Calculate parameters of conditional distribution
        loc = self.theta*x1
        scale = np.sqrt((self.df + x1**2)/(self.df + 1)*(1 - self.theta**2))
        df = self.df + 1

        d2 = stats.t(loc=loc,scale=np.sqrt(scale),df=df)

        return d2.cdf(d1.ppf(v))

    def inv_cdf_v_given_u(self,u,p):
        """
        param: u: uniform random variable on [0,1]
        param: p: probability that V ≤ v given U = u
        returns: v: value of v such that Pr(V ≤ v | U = u) = p
        """
        d1 = stats.t(df=self.df)
        x1 = d1.ppf(u)

        # Calculate parameters of conditional distribution
        loc = self.theta*x1
        scale = np.sqrt((self.df + x1**2)/(self.df + 1)*(1 - self.theta**2))
        df = self.df + 1

        d2 = stats.t(loc=loc,scale=scale,df=df)

        return d1.cdf(d2.ppf(p))

    def draw_v_given_u(self,u):
        """
        param: u: specified values of u
        """
        p = stats.uniform(loc=0,scale=1).rvs(size=u.shape)
        v = self.inv_cdf_v_given_u(u,p)
        return(v)

    def draw_uv_pairs(self,n):
        """
        param: n: number of (u,v) pairs to draw
        """
        u = stats.uniform(loc=0,scale=1).rvs(size=n)
        v = self.draw_v_given_u(u)
        return(u,v)

    def kendalls_tau(self):
        """
        returns: tau_k: kendall rank correlation coefficient
        """
        tau_k = 2/np.pi*np.arcsin(self.theta)
        return(tau_k)

    def spearmans_rho(self):
        """
        returns: rho_s: spearman rank correlation coefficient
        """
        rho_s = 6/np.pi*np.arcsin(self.theta/2)
        return(rho_s)

class ArchimedeanCopula:

    """
    Wrapper class for bivariate archimedean copulas that allows for rotation and negative depedence
    """
    def __init__(self,theta,family,rotation=0):
        """
        param: theta: dependence parameter
        param: family: name of specific archimedean copula used to model dependence
        param: rotation: modification of copula by rotating it by either 0, 90, 180, or 270 degrees
        """
        family_options = ['Clayton','Frank','Gumbel','Joe']
        rotation_options = [0,90,180,270]

        # Check that user input is in list of availabile archimedean copulas
        if family not in family_options:
            raise ValueError(f'Unrecognized copula family \'{family}\'. Must be one of '+', '.join(str(cf) for cf in family_options))
        else:
            family = 'Bivariate' + family

        # Check that rotation is one of 0, 90, 180, 270
        if rotation not in rotation_options:
            raise ValueError(f'Unrecognized rotation \'{rotation}\'. Must be one of '+', '.join(str(r) for r in rotation_options))

        self.theta=theta
        self.family=family
        self.rotation=rotation
        self.copula = getattr(sys.modules[__name__],family)(theta=theta)

        # Check that value of theta is within supported range
        support = self.copula.support['theta']

        # If value of theta is on edge of range, adjust by small number to avoid numerical problems down the road
        smallnum = np.finfo(float).eps

        if (theta < support[0]) or (theta > support[1]):
            raise ValueError(f'Parameter theta must be between {support[0]} and {support[1]} for family {family}.')
        elif (theta == support[0]):
            self.copula = getattr(sys.modules[__name__],family)(theta=theta+smallnum)
        elif (theta == support[1]):
            self.copula = getattr(sys.modules[__name__],family)(theta=theta-smallnum)

        # Specify mathematical functions for rotated copula as function of those for unrotated copula

        if rotation == 0:
            self.pdf = lambda u,v: self.copula.pdf(u,v)
            self.cdf = lambda u,v: self.copula.cdf(u,v)
            self.cdf_v_given_u = lambda u,v: self.copula.cdf_v_given_u(u,v)
            self.inv_cdf_v_given_u = lambda u,p: self.copula.inv_cdf_v_given_u(u,p)
            self.kendalls_tau = lambda: self.copula.kendalls_tau()
            self.spearmans_rho = lambda: self.copula.spearmans_rho()

        elif rotation == 90:
            self.pdf = lambda u,v: self.copula.pdf(u,1-v)
            self.cdf = lambda u,v: self.copula.cdf(u,1) - self.copula.cdf(u,1-v)
            self.cdf_v_given_u = lambda u,v: self.copula.cdf_v_given_u(u,1) - self.copula.cdf_v_given_u(u,1-v)
            self.inv_cdf_v_given_u = lambda u,p: 1 - self.copula.inv_cdf_v_given_u(u,1-p)
            self.kendalls_tau = lambda: -1*self.copula.kendalls_tau()
            self.spearmans_rho = lambda: -1*self.copula.spearmans_rho()

        elif rotation == 180:
            self.pdf = lambda u,v: self.copula.pdf(1-u,1-v)
            self.cdf = lambda u,v: 1 - self.copula.cdf(1-u,1) - self.copula.cdf(1,1-v) + self.copula.cdf(1-u,1-v)
            self.cdf_v_given_u = lambda u,v: self.copula.cdf_v_given_u(1-u,1) - self.copula.cdf_v_given_u(1-u,1-v)
            self.inv_cdf_v_given_u = lambda u,p: 1 - self.copula.inv_cdf_v_given_u(1-u,1-p)
            self.kendalls_tau = lambda: self.copula.kendalls_tau()
            self.spearmans_rho = lambda: self.copula.spearmans_rho()

        else:
            self.pdf = lambda u,v: self.copula.pdf(1-u,v)
            self.cdf = lambda u,v: self.copula.cdf(1,v) - self.copula.cdf(1-u,v)
            self.cdf_v_given_u = lambda u,v: self.copula.cdf_v_given_u(1-u,v)
            self.inv_cdf_v_given_u = lambda u,p: self.copula.inv_cdf_v_given_u(1-u,p)
            self.kendalls_tau = lambda: -1*self.copula.kendalls_tau()
            self.spearmans_rho = lambda: -1*self.copula.spearmans_rho()

    def draw_v_given_u(self,u):
        """
        param: u: specified values of u
        """
        p = stats.uniform(loc=0,scale=1).rvs(size=u.shape)
        v = self.inv_cdf_v_given_u(u,p)
        return(v)

    def draw_uv_pairs(self,n):
        """
        param: n: number of (u,v) pairs to draw
        """
        u = stats.uniform(loc=0,scale=1).rvs(size=n)
        v = self.draw_v_given_u(u)
        return(u,v)
    
    def log_likelihood(self,u,v,weights=None):
        """
        param: u: specified values of u
        param: v: specified values of v
        param: weights: vector of weights assigned to each (u,v) pair
        returns: value of log-likelihood function
        """
        if weights is None:
            weights=np.ones(len(u))
        
        LL = np.sum(weights*np.log(self.pdf(u,v)))
        return(LL)

class GaussianCopula(BivariateGaussian):
    """
    Child class for bivariate gaussian copula that includes some checks for user input
    """
    def __init__(self,theta):

        smallnum = np.finfo(float).eps

        if (theta < -1) or (theta > 1):
            raise ValueError('Parameter theta must be between -1 and 1.')
        elif theta == -1:
            theta = theta + smallnum
        elif theta == 1:
            theta = theta - smallnum

        # Call initialization procedures of parent class
        BivariateGaussian.__init__(self, theta)
        
    def log_likelihood(self,u,v,weights=None):
        """
        param: u: specified values of u
        param: v: specified values of v
        param: weights: vector of weights assigned to each (u,v) pair
        returns: value of log-likelihood function
        """
        if weights is None:
            weights=np.ones(len(u))
        
        LL = np.sum(weights*np.log(self.pdf(u,v)))
        return(LL)

class StudentsTCopula(BivariateStudentsT):
    """
    Child class for bivariate t-copula that includes some checks for user input
    """
    def __init__(self,theta,df):

        smallnum = np.finfo(float).eps

        if (theta < -1) or (theta > 1):
            raise ValueError('Parameter theta must be between -1 and 1.')
        elif theta == -1:
            theta = theta + smallnum
        elif theta == 1:
            theta = theta - smallnum

        if (df < 0):
            raise ValueError('Parameter df must be greater than zero')
        elif df == 0:
            df = df+smallnum

        # Call initialization procedures of parent class
        BivariateStudentsT.__init__(self, theta, df)
        
    def log_likelihood(self,u,v,weights=None):
        """
        param: u: specified values of u
        param: v: specified values of v
        param: weights: vector of weights assigned to each (u,v) pair
        returns: value of log-likelihood function
        """
        if weights is None:
            weights=np.ones(len(u))
        
        LL = np.sum(weights*np.log(self.pdf(u,v)))
        return(LL)
    
class empirical_distribution:
    """
    Class for modeling the empirical distribution of a random variable based on samples of data. 
    """
    
    def __init__(self,x,weights=None):
        """
        param: x: sampled values of x
        param: weights: weights associated with sampled values of x
        """
    
        if weights is None:
            weights = np.ones(len(x))
            
        # Remove nans
        mask = ~np.isnan(x)
        x = x[mask]
        weights = weights[mask]
        
        sort_inds = np.argsort(x)
        x = x[sort_inds]
        weights = weights[sort_inds]

        # For weighted CDF, Pr(X <= x) = sum(weights[X <= x])
        F = np.cumsum(weights)/np.sum(weights)

        self.cdf = interp.interp1d(x,F,kind='linear',bounds_error=False,fill_value=(0,1))
        self.ppf = interp.interp1d(F,x,kind='linear',bounds_error=False,fill_value=(np.min(x),np.max(x)))
        self.kde = stats.gaussian_kde(x,weights=weights)
        self.pdf = np.vectorize(self.kde.pdf)
        
    def rvs(self,n=1):
        """
        Randomly draw values from empirical distribution using inverse transform sampling
        """
        return self.ppf(stats.uniform().rvs(n))
        
def fit_archimedean_copula(u,v,family_options=['Clayton','Frank','Gumbel','Joe'],rotation_options=[0,90,180,270],weights=None):
    """
    Determine the best-fit Archimedean copula based on maximum likelihood estimation.
    
    param: u: specified values of u
    param: v: specified values of v
    param: weights: vector of weights assigned to each (u,v) pair
    """
    
    smallnum = 1e-6
    family_support = {'Clayton':[0,50],'Frank':[0,25],'Gumbel':[1,50],'Joe':[1,50]}
    
    best_family = family_options[0]
    best_rotation = rotation_options[0]
    best_likelihood = -np.inf
    best_theta = None
    
    for i,family in enumerate(family_options):
        for rotation in rotation_options:
            
            bounds = family_support[family]
            bounds[0] = bounds[0] + smallnum
            bounds[1] = bounds[1] - smallnum
            initial_guess = np.mean(bounds)
            
            # Find theta that maximizes log-likelihood for given copula
            obj_fun = lambda theta: -1*ArchimedeanCopula(theta=theta,family=family,rotation=rotation).log_likelihood(u,v,weights=weights)
            res = so.minimize_scalar(obj_fun,initial_guess,bounds=bounds)
            
            theta_fit = res.x
            LL = -1*res.fun
                        
            if LL > best_likelihood:
                            
                best_family = family
                best_rotation = rotation
                best_likelihood = LL
                best_theta = theta_fit
                
    print(f'Best fit Archimedean copula: {best_family}(theta={np.round(best_theta,2)}, rotation={best_rotation})')
    print(f'Log-likelihood: {np.round(best_likelihood,2)}\n')
    c = ArchimedeanCopula(theta=best_theta,family=best_family,rotation=best_rotation)
    extra = [best_likelihood,best_theta,best_family,best_rotation]
    return(c,extra)

def fit_gaussian_copula(u,v,weights=None):
    """
    Fit a gaussian copula based on maximum likelihood estimation.
    
    param: u: specified values of u
    param: v: specified values of v
    param: weights: vector of weights assigned to each (u,v) pair
    """
    smallnum = 1e-6
    bounds = [-1+smallnum,1-smallnum]
    d = stats.norm()
    initial_guess = stats.pearsonr(d.ppf(u),d.ppf(v)).statistic
    
    obj_fun = lambda theta: -1*GaussianCopula(theta).log_likelihood(u,v,weights=weights)
    res = so.minimize_scalar(obj_fun,initial_guess,bounds=bounds)
            
    theta_fit = res.x
    LL = -1*res.fun
    
    print(f'Best fit Gaussian copula: GaussianCopula(theta={np.round(theta_fit,2)})')
    print(f'Log-likelihood: {np.round(LL,2)}\n')
    c = GaussianCopula(theta_fit)
    extra = [LL,theta_fit]
    return(c,extra)

class MultivariateGaussianCopula:
    """
    Multivariate gaussian copula class - supports arbitrary number of variables
    """
    def __init__(self,R):
        """
        param: R: correlation matrix (d x d array)
        """
        # Check that R is a valid correlation matrix
        cond1 = sla.issymmetric(R)
        cond2 = np.allclose(np.diag(R),1)
        cond3 = (np.max(np.abs(R)) <= 1)
            
        if not cond1&cond2&cond3:
            raise ValueError('Matrix R is not a valid correlation matrix.')
        
        self.R = R
        
        # Extra attributes specific to gaussian distribution
        self.mv_norm = stats.multivariate_normal(cov=R)
        
    def pdf(self,uu):
        """
        param: uu: array of m realizations of d uniform random variables (m x d array)
        returns: probability density associated with each realization
        """
        d = stats.norm()
        zz = d.ppf(uu)
        smallnum = np.finfo(float).eps
        numerator = self.mv_norm.pdf(zz)
        denominator = np.exp(np.sum(d.logpdf(zz),axis=1))
        return np.maximum(numerator/(denominator+smallnum),smallnum)

    def cdf(self,uu):
        """
        param: uu: array of m realizations of d uniform random variables (m x d array)
        returns: probability that U1 ≤ u1, U2 ≤ u2, ..., Ud ≤ ud for each of the m realizations
        """
        d = stats.norm()
        zz = d.ppf(uu)
        return self.mv_norm.cdf(zz)
    
    def simulate_values(self,n):
        """
        Randomly draw values from parameterized multivariate gaussian copula. 
        
        param: n: number of realizations to simulate
        returns: uu: array of n simulated realizations of d uniform random variables (n x d array)
        """
        d = stats.norm()
        zz = self.mv_norm.rvs(size=n)
        uu = d.cdf(zz)
        return(uu)
    
    def conditional_distribution(self,u):
        """
        Given the depedence structure of the gaussian copula and known values of variables, determine the 
        multivariate gaussian distribution that characterizes the conditional copula of unknown variables.
        
        See: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        
        param: u: one realization of d uniform random variables (1 x d array) with unknown values represented by np.nan 
        returns: cond_mv_norm: multivariate guassian distribution of unknown vars conditioned on known vars
        returns: known_cols: column indicies associated with known vars
        returns: unknown_cols: column indices associated with unknown vars
        """
        known_cols = np.where(~np.isnan(u))[0]
        unknown_cols = np.where(np.isnan(u))[0]
        n_known = len(known_cols)
        n_unknown = len(unknown_cols)
        
        d = stats.norm()
        z_known = d.ppf(u[known_cols])

        R11 = self.R[unknown_cols,:][:,unknown_cols]
        R22 = self.R[known_cols,:][:,known_cols]
        R12 = self.R[unknown_cols,:][:,known_cols]
        R21 = R12.T

        R22_inv = np.linalg.inv(R22)

        mu = R12 @ R22_inv @ z_known
        sigma = R11 - R12 @ R22_inv @ R21

        cond_mv_norm = stats.multivariate_normal(mean=mu,cov=sigma)
        
        return(cond_mv_norm,known_cols,unknown_cols)
    
    def conditional_simulation(self,uu):
        """
        param: uu: array of m realizations of d uniform random variables (m x d array) with unknown values represented by np.nan
        returns: uu_sim: input array with values of unknown variables simulated conditional on values of known variables 
        """
        d = stats.norm()
        uu_sim = uu.copy()
        
        for i in range(uu_sim.shape[0]):
            
            # Get gaussian distribution conditional on realization of u
            u = uu_sim[i]
            
            if np.isnan(u).any():
                
                cond_mv_norm,known_cols,unknown_cols = self.conditional_distribution(u)

                # Simulate from conditoinal gaussian distribution
                z_cond_sim = cond_mv_norm.rvs()

                # Transform back to a uniform variable
                u_cond_sim = d.cdf(z_cond_sim)
                uu_sim[i,unknown_cols] = u_cond_sim
                
            else:
                uu_sim[i] = u
            
        return(uu_sim)        
    
class DependenceModel:
    """
    Class used to model the joint distribution of random variables.
    The dependence between variables is modeled using a multivariate gaussian copula. 
    """
    
    def __init__(self,marginals):
        """
        param: marginals: dict of marginal distributions. 
                          Each key is the name of the variable. 
                          Each value is an instance of scipy.stats.rv_continuous (or a similar class)
        """
        self.n_vars = len(marginals)
        self.var_names = list(marginals.keys())
        self.name_to_index = {name:i for i,name in enumerate(self.var_names)}
        self.var_dists = list(marginals.values())
        self.marginals = marginals
        
        # At initialization, assume everything is independent
        self.R = np.identity(self.n_vars)
        self.copula = MultivariateGaussianCopula(self.R)
        
    def add_dependence(self,var1,var2,theta):
        """
        Add dependence/correlation between two variables. This will be modeled using a gaussian copula. 
        
        param: var1: name of variable #1
        param: var2: name of variable #2
        param: theta: value of dependence parameter between variables 1 & 2
        """
        if var1 not in self.var_names:
            raise ValueError(f'Unrecognized variable name \'{var1}\'')
        if var2 not in self.var_names:
            raise ValueError(f'Unrecognized variable name \'{var2}\'')
        if var1 == var2:
            raise ValueError(f'A variable cannot have dependence with itself.')
        if np.abs(theta) > 1:
            raise ValueError(f'Theta must be between -1 and 1.')
            
        i = self.name_to_index[var1]
        j = self.name_to_index[var2]
            
        self.R[i,j] = theta
        self.R[j,i] = theta
        self.copula = MultivariateGaussianCopula(self.R)
        
    def fit_dependence(self,df):
        """
        Fit a multiavariate gaussian copula to observed data. 
        param: df: dataframe of observed realizations of variables. Column names should be the same as variable names. 
        """
        included_vars = [var for var in self.var_names if var in df.columns]
        df = df[included_vars]
        
        for var1,var2 in itertools.combinations(included_vars,2):
            
            print(f'*** Fitting dependence between {var1} and {var2} ***',flush=True)
            
            mask = (~df[var1].isna())&(~df[var2].isna())
            
            x1 = df[mask][var1].to_numpy()
            x2 = df[mask][var2].to_numpy()
            u1 = self.marginals[var1].cdf(x1)
            u2 = self.marginals[var2].cdf(x2)
            
            c,extra = fit_gaussian_copula(u1,u2)
            theta = extra[-1]
            
            self.add_dependence(var1,var2,theta)
    
    def pdf(self,xx):
        """
        Return the probability density of a multivariate distribution
        
        Calculated based on the density of the copula and the marginals based on Sklar's theorem. 
        See: https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem
        
        param: xx: m realizations of d random variables (m x d array)
        returns: probability density associated with each realization
        """
        ff = np.zeros(xx.shape)
        uu = np.zeros(xx.shape)
        for i in range(self.n_vars):
            ff[:,i] = self.var_dists[i].pdf(xx[:,i])
            uu[:,i] = self.var_dists[i].cdf(xx[:,i])
                
        return self.copula.pdf(uu)*np.prod(ff,axis=1)
    
    def cdf(self,xx):
        """
        param: xx: m realizations of d random variables (m x d array)
        returns: probability that X1 ≤ x1, X2 ≤ x2, ..., Xd ≤ xd for each of the m realizations
        """
        uu = np.zeros(xx.shape)
        for i in range(self.n_vars):
            uu[:,i] = self.var_dists[i].cdf(xx[:,i])
            
        return self.copula.cdf(uu)
    
    def simulate_values(self,n,return_dataframe=False):
        """
        Randomly draw values from multivariate distribution. 
        
        param: n: number of realizations to simulate
        param: return_dataframe: if true, return a pandas dataframe of simulated variables instead of numpy array
        returns: xx: array of n simulated realizations of d random variables (n x d array)
        """
        uu = self.copula.simulate_values(n)
        xx = np.zeros(uu.shape)
        
        for i in range(self.n_vars):
            xx[:,i] = self.var_dists[i].ppf(uu[:,i])
            
        if return_dataframe:
            return pd.DataFrame(xx,columns=self.var_names)
        else:
            return xx
        
    def conditional_simulation(self,xx,return_dataframe=False):
        """
        param: xx: array of m realizations of d random variables (m x d array) with unknown values represented by np.nan
        param: return_dataframe: if true, return a pandas dataframe of simulated variables instead of numpy array
        returns: xx_sim: input array with values of unknown variables simulated conditional on values of known variables 
        """
        uu = np.zeros(xx.shape)
        for i in range(self.n_vars):
            uu[:,i] = self.var_dists[i].cdf(xx[:,i])
            
        uu_sim = self.copula.conditional_simulation(uu)
        
        xx_sim = np.zeros(uu_sim.shape)
        for i in range(self.n_vars):
            xx_sim[:,i] = self.var_dists[i].ppf(uu_sim[:,i])
        
        if return_dataframe:
            return pd.DataFrame(xx_sim,columns=self.var_names)
        else:
            return xx_sim        
        
        
        
        
        
        
            
        
        
        
            
            
            