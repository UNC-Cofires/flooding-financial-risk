import numpy as np
import scipy.stats as stats
import scipy.integrate as si
import scipy.optimize as so
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
        return self.multivariate_normal_pdf(x,y)/(d.pdf(x)*d.pdf(y))

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
        return self.multivariate_t_pdf(x,y)/(d.pdf(x)*d.pdf(y))

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
    
def best_fit_archimedean_copula(u,v,weights=None):
    """
    Determine the best-fit Archimedean copula based on maximum likelihood estimation.
    
    param: u: specified values of u
    param: v: specified values of v
    param: weights: vector of weights assigned to each (u,v) pair
    """
    
    smallnum = 1e-6
    family_options = ['Clayton','Frank','Gumbel','Joe']
    family_support = [[0,50],[0,25],[1,50],[1,50]]
    rotation_options = [0,90,180,270]
    
    best_family = family_options[0]
    best_rotation = rotation_options[0]
    best_likelihood = -np.inf
    best_theta = None
    
    for i,family in enumerate(family_options):
        for rotation in rotation_options:
            
            bounds = family_support[i]
            bounds[0] = bounds[0] + smallnum
            bounds[1] = bounds[1] - smallnum
            initial_guess = np.mean(bounds)
            
            # Find theta that maximizes log-likelihood for given copula
            obj_fun = lambda theta: -1*ArchimedeanCopula(theta=theta,family=family,rotation=rotation).log_likelihood(u,v,weights=weights)
            res = so.minimize_scalar(obj_fun,initial_guess,bounds=bounds)
            
            theta_fit = res.x
            LL = -1*res.fun
            
            #print(f'{family} {rotation} {np.round(theta_fit,2)} {np.round(LL,2)}')
            
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

def best_fit_elliptical_copula(u,v,weights=None):
    """
    Determine the best-fit elliptical copula based on maximum likelihood estimation.
    
    param: u: specified values of u
    param: v: specified values of v
    param: weights: vector of weights assigned to each (u,v) pair
    """
    smallnum = 1e-6
    
    # First try fitting gaussian copula
    bounds = [-1+smallnum,1-smallnum]
    initial_guess = 0.0

    # Find theta that maximizes log-likelihood for given copula
    obj_fun = lambda theta: -1*GaussianCopula(theta).log_likelihood(u,v,weights=weights)
    res = so.minimize_scalar(obj_fun,initial_guess,bounds=bounds)

    theta_fit = res.x
    LL = -1*res.fun
    
    best_family = 'Gaussian'
    best_df = None
    best_likelihood = LL
    best_theta = theta_fit
    
    # Also try Student's t-distribution
    # Note that variance undefined for df < 3
    # And anything with df >= 30 is approximately gaussian
    # If no noticable improvement by reducing df, stop process early
    
    num_failures = 0
    
    for df in range(29,2,-1):
        
        # Find theta that maximizes log-likelihood for given copula
        obj_fun = lambda theta: -1*StudentsTCopula(theta,df).log_likelihood(u,v,weights=weights)
        res = so.minimize_scalar(obj_fun,initial_guess,bounds=bounds)
            
        theta_fit = res.x
        LL = -1*res.fun
                    
        if LL > best_likelihood:
            best_family = 'StudentsT'            
            best_df = df
            best_likelihood = LL
            best_theta = theta_fit
            
            num_failures = 0
        else:
            num_failures += 1
            
        if num_failures >= 5:
            break
    
    if best_family == 'Gaussian':
        print(f'Best fit elliptical copula: {best_family}(theta={np.round(best_theta,2)})')
        print(f'Log-likelihood: {np.round(best_likelihood,2)}\n')
        c = GaussianCopula(best_theta)
    else:
        print(f'Best fit elliptical copula: {best_family}(theta={np.round(best_theta,2)}, df={best_df})')
        print(f'Log-likelihood: {np.round(best_likelihood,2)}\n')
        c = StudentsTCopula(best_theta,best_df)
        
    extra = [best_likelihood,best_theta,best_family,best_df]
            
    return(c,extra)

def best_fit_copula(u,v,weights=None):
    """
    Determine the best-fit copula based on maximum likelihood estimation. 
    
    param: u: specified values of u
    param: v: specified values of v
    param: weights: vector of weights assigned to each (u,v) pair
    """
    
    c1,extra1 = best_fit_archimedean_copula(u,v,weights=weights)
    c2,extra2 = best_fit_elliptical_copula(u,v,weights=weights)
    
    if extra1[0] >= extra2[0]:
        c = c1
        extra = extra2
        best_likelihood,best_theta,best_family,best_rotation = extra1
        print(f'Best fit copula: {best_family}(theta={np.round(best_theta,2)}, rotation={best_rotation})')
        print(f'Log-likelihood: {np.round(best_likelihood,2)}\n')
    else:
        c = c2
        extra = extra2
        best_likelihood,best_theta,best_family,best_df = extra2
        if best_family == 'StudentsT':
            print(f'Best fit copula: {best_family}(theta={np.round(best_theta,2)}, df={best_df})')
        else:
            print(f'Best fit copula: {best_family}(theta={np.round(best_theta,2)})')
            
        print(f'Log-likelihood: {np.round(best_likelihood,2)}\n')
        
    return(c,extra)