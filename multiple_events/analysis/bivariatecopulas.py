import numpy as np
import scipy.stats as stats
import scipy.integrate as si
import scipy.optimize as so

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
        return self.multivariate_normal_pdf(d.ppf(u),d.ppf(v))

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
        return self.multivariate_t_pdf(d.ppf(u),d.ppf(v))

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
