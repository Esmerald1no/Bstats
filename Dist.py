from scipy import stats as st
import numpy as np

class dist:
    
    def __init__(self,obj = None,type:str = None,name:str = "",**kwargs) -> None:
        self.type = type
        self.name = name
        match type:
            case "gaussian":
                self.mu = kwargs.get("mu")
                self.sigma = kwargs.get("sigma")

                self.dist = self.g_dist(self.mu,self.sigma).gdist

            case "normal":
                self.mu = 1
                self.sigma = 0

                self.dist = self.g_dist(self.mu,self.sigma).gdist
            
            case "beta":
                self.alpha = kwargs.get("alpha")
                self.beta = kwargs.get("beta")

                self.dist= self.b_dist(self.alpha,self.beta).bdist
            
            case "bivar_gaussian":
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
        def __init__(self,alpha,beta) -> None:
            self.bdist = st.beta(alpha,beta)

    class bivar_g_dist:
        def __init__(self,a_mean,b_mean,a_var,b_var,corr) -> None:
            self.bivar_gdist = st.multivariate_normal(mean=np.array([a_mean,b_mean]),cov=dist.cp_cov_mat(a_var,b_var,corr))

    class paired_dist:
        def __init__(self,a,b) -> None:
            self.paired_dist = np.array([j-i for i,j in zip(a,b)])
            

    @staticmethod
    def cp_cov_mat(a_var,b_var,corr):
        return np.array([[a_var,corr*np.sqrt(a_var*b_var)],[corr*np.sqrt(a_var*b_var),b_var]])

    def cp_mean(self):
        return np.mean(self.dist)
    
    def cp_median(self):
        return np.median(self.dist)
    
    def cp_mode(self):
        return st.mode(self.dist,keepdims=False)[0]
    
    def cp_quartiles(self):
        return np.percentile(self.dist,[0,25,50,75,100])
    
    def cp_iqr(self):
        return st.iqr(self.dist)
    
    def cp_range(self):
        return np.ptp(self.dist)
    
    def cp_var(self):
        return st.tvar(self.dist)

    def cp_std(self):
        return st.tstd(self.dist)

    def cp_std_err(self):
        return st.tsem(self.dist)

    def cp_coef_var(self):
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