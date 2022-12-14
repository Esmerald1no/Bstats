from scipy import stats as st
import numpy as np

class NotAvailableError(Exception):
    """Raised when using an unsupported function on a Continuous Distrinution"""
    pass

class dist:
    
    def __init__(self,obj = None,type:str = None,name:str = "",**kwargs) -> None:
        self.type = type.strip().lower()
        self.name = name
        match self.type:
            case "gaussian" | "g":
                assert "mu" in kwargs, "'mu'(Mean) must be provided as an agrument."
                assert "sigma" in kwargs, "'sigma'(Standard Deviation) must be provided as an argument."

                self.mu = kwargs.get("mu")
                self.sigma = kwargs.get("sigma")

                self.dist = self.g_dist(self.mu,self.sigma).gdist

            case "normal":
                self.mu = 1
                self.sigma = 0

                self.dist = self.g_dist(self.mu,self.sigma).gdist
            
            case "beta":
                assert "alpha" in kwargs, "'alpha' must be provided as an agrument."
                assert "beta" in kwargs, "'beta' must be provided as an argument."

                self.alpha = kwargs.get("alpha")
                self.beta = kwargs.get("beta")

                self.dist= self.b_dist(self.alpha,self.beta).bdist
            
            case "bivar_gaussian":
                assert "a_mean" in kwargs, "'a_mean' (Mean of Distribution A) must be provided as an agrument."
                assert "b_mean" in kwargs, "'b_mean' (Mean of Distribution B) must be provided as an argument."
                assert "a_var" in kwargs, "'a_var' (Variance of Distribution A) must be provided as an agrument."
                assert "b_var" in kwargs, "'b_var' (Variance of Distribution B) must be provided as an argument."
                assert "corr" in kwargs, "'corr' (Correlation Coefficient) must be provided as an argument "

                self.a_mean = kwargs.get("a_mean")
                self.b_mean = kwargs.get("b_mean")
                self.a_var = kwargs.get("a_var")
                self.b_var = kwargs.get("b_var")
                self.corr = kwargs.get("corr")
                
                self.dist = self.bivar_g_dist(self.a_mean,self.b_mean,self.a_var,self.b_var,self.corr).bivar_gdist

            case "paired":
                a = kwargs.get("a")
                b = kwargs.get("b")

                self.dist = self.paired_dist(a,b).paired_dist
                self.count = self.dist.size

            case _:
                self.dist = np.array(obj)
                self.count = self.dist.size

    class g_dist:
        def __init__(self,mu:float,sigma:float) -> None:
            self.gdist = st.norm(mu,sigma)


    class b_dist:
        def __init__(self,alpha:float,beta:float) -> None:
            self.bdist = st.beta(alpha,beta)

    class bivar_g_dist:
        def __init__(self,a_mean:float,b_mean:float,a_var:float,b_var:float,corr:float) -> None:
            self.bivar_gdist = st.multivariate_normal(mean=np.array([a_mean,b_mean]),cov=dist.cp_cov_mat(a_var,b_var,corr))

    class paired_dist:
        def __init__(self,a:np.ndarray,b:np.ndarray) -> None:
            self.paired_dist = np.array([j-i for i,j in zip(a,b)])
            

    @staticmethod
    def cp_cov_mat(a_var,b_var,corr):
        return np.array([[a_var,corr*np.sqrt(a_var*b_var)],[corr*np.sqrt(a_var*b_var),b_var]])

    @staticmethod
    def sample(dist_type:str, size:int, **dist_params):
        #Prefered sampling method over dist.rvs()
        #Use this method if you are sampling often or using while/for loops

        match dist_type.lower().strip():
            case ("gaussian"|"normal"):
                return st.norm.rvs(dist_params.get("mu"), dist_params.get("sigma"), size = size)
            case "beta":
                return st.beta.rvs(dist_params.get("alpha"), dist_params.get("beta"), size = size)
            case "gamma":
                return st.gamma.rvs(dist_params.get("alpha"), dist_params.get("beta"), size = size)
            case ("bivar_gaussian"|"bivar_normal"):
                a_mean = dist_params.get("a_mean")
                b_mean = dist_params.get("b_mean")
                a_var = dist_params.get("a_var")
                b_var = dist_params.get("b_var")
                corr = dist_params.get("corr")
                return st.multivariate_normal.rvs(mean=[a_mean,b_mean],cov=dist.cp_cov_mat(a_var,b_var,corr), size = size)

    #Methods Abaliable for Continuous Distributions and Arrays:

    @property
    def cp_mean(self):
        if type(self.dist) == np.ndarray:
            return np.mean(self.dist)
        else:
            return self.dist.mean()
    
    @property
    def cp_median(self):
        if type(self.dist) == np.ndarray:
            return np.median(self.dist)
        else:
            return self.dist.median()

    @property
    def cp_std(self):
        if type(self.dist) == np.ndarray:
            return st.tstd(self.dist)
        else:
            return self.dist.std()
    
    #Methods NOT available to Continuous Distributions, only Arrays.

    @property
    def cp_mode(self):
        if type(self.dist) == np.ndarray: raise NotAvailableError
        return st.mode(self.dist,keepdims=False)[0]
    
    @property
    def cp_quartiles(self):
        if type(self.dist) == np.ndarray: raise NotAvailableError
        return np.percentile(self.dist,[0,25,50,75,100])

    @property
    def cp_iqr(self):
        if type(self.dist) == np.ndarray: raise NotAvailableError
        return st.iqr(self.dist)
    
    @property
    def cp_range(self):
        if type(self.dist) == np.ndarray: raise NotAvailableError
        return np.ptp(self.dist)
    
    @property
    def cp_var(self):
        if type(self.dist) == np.ndarray: raise NotAvailableError
        return st.tvar(self.dist)

    @property
    def cp_std_err(self):
        if type(self.dist) == np.ndarray: raise NotAvailableError
        return st.tsem(self.dist)

    @property
    def cp_coef_var(self):
        if type(self.dist) == np.ndarray: raise NotAvailableError
        return st.variation(self.dist,ddof=1)

    def compute_internals(self):
        self.mean = self.cp_mean()
        self.median = self.cp_median()
        self.mode = self.cp_mode()
        self.quartiles = self.cp_quartiles()
        self.iqr = self.cp_iqr()
        self.range = self.cp_range()
        self.variance = self.cp_var()
        self.std = self.cp_std()
        self.st_err = self.cp_std_err()
        self.coef_var = self.cp_coef_var()


    def report(self):
        self.compute_internals()
        print(f'''Report of Distribution:
Mean : {self.cp_mean():.4f}
Median: {self.median}
Mode: {self.mode}
Range: {self.range}
Quartiles(0%, 25%, 50%, 75%, 100%): {[str(i) for i in self.quartiles]}
IQR: {self.iqr}
Variance: {self.variance:.4f}
Standard Deviation: {self.std:.4f}
Standard Error: {self.st_err:.4f}
Coefficient of Variation: {self.coef_var:.4f}
               '''
            )