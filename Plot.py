import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm


class Multi_Plot:
    def __init__(self,num_rows,num_cols,**kwargs) -> None:
        self.fig, self.axs = plt.subplots(ncols=num_cols,nrows=num_rows,**kwargs)
        
        self.curr_row = 0
        self.curr_col = 0

        self.row_max = num_rows
        self.col_max = num_cols
    
    def inc_row(self):
        if self.curr_row +1 > self.row_max:
            raise IndexError("Figure too small, increase number of rows.")
    
        self.curr_row +=1
    
    def set_row(self,val):
        if abs(val) > self.row_max:
            raise IndexError("Index out of range.")

        self.curr_row = val
    
    def inc_col(self):
        if self.curr_col +1 > self.col_max:
            raise IndexError("Figure too small, increase number of columns.")
        
        self.curr_col +=1
    
    def set_col(self,val):
        if abs(val) > self.col_max:
            raise IndexError("Index out of range.")

        self.curr_col = val

    def ret_axs_index(self):

        try:
            if self.curr_col < self.col_max:
                self.inc_col()
            elif self.curr_row < self.row_max:
                self.inc_row()
                self.curr_col = 0
            else:
                self.curr_row = self.row_max
                self.curr_col = self.col_max

        except IndexError:
            print("Figure is Full.")
        else:
            return self.curr_row, self.curr_col-1
    
        
        
        #TODO: I need to return the current index so that the plotting functions 
        #know where to write the figure to in the subplot, however, I also don't 
        #want to manually increase the counters, so I need a way to return the current
        #value and then increment the counter without raising the Out of Index Exeption. 

        #see if code beraks?

    def ret_ax(self):
        row,col = self.ret_axs_index()
        
        if self.row_max == 1:
            return self.axs[col]
        else:
            return self.axs[row,col]

        #TODO: TypeError: Axes.hist() got multiple values for argument 'ax' not sure what is jappening here.


#TODO: Add the "return_fig = False" argument to all plotting function, alognside the "ax" optional
# argument so that the function can now add the output figure to the Multi_Plot. 


def histogram(dist,bins:str = None,title:str = "TITLE",x_axis:str = "X Axis",y_axis:str = "Y Axis", return_fig = False,**kwargs):
    if bins != None:
        bins = "auto"

    if return_fig:
        ax_i = kwargs.get("ax")
        ax_i.hist(dist,bins, facecolor="b", alpha = 0.5, ec ="black",**kwargs)
    
    else:
        _n, _bin,_patches = plt.hist(dist,bins, facecolor="b", alpha = 0.5, ec ="black",**kwargs)

        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

        plt.show()


def boxplot(dist,*,title:str = "",x_axis:str = "",y_axis:str = "",scatter=False,widths=0.5,**kwargs):
    if type(dist) != list :
        _dict = plt.boxplot(dist,widths=widths,**kwargs)
    else:
        _dict = plt.boxplot(x=[dst.dist for dst in dist],widths=widths)
        plt.xticks(ticks=range(1,len(dist)+1),labels=[dst.name for dst in dist])
    
    if scatter:
        for i in range(len(dist)):
            plt.plt.scatter(i+1 + plt.np.random.random(dist[i].count) * widths/2 -widths/4, dist[i].dist,s=8,edgecolors="black",facecolors='none')

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()


def plot(x,y,title:str = "",x_axis:str = "x",y_axis:str = "y",**kwargs):

    plt.plot(x,y,**kwargs)

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()


def plot_pdf(dist,title:str = "",x_axis:str = "x",y_axis:str = "Probability",**kwargs):
    x = np.linspace(dist.cp_mean() - 4*dist.cp_std(), dist.cp_mean() + 4*dist.cp_std(), 100)

    plt.plot(x, dist.pdf(x),**kwargs)

    plt.gca().set_ylim([0,None])

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()


def plot_cdf(dist,title:str = "",x_axis:str = "",y_axis:str = "Probability",**kwargs):
    x = np.linspace(dist.cp_mean() - 4*dist.cp_std(), dist.cp_mean() + 4*dist.cp_std(), 100)

    plt.plot(x, dist.cdf(x),**kwargs)

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()


def plot_qq(dist,line = "s",title:str = "",x_axis:str = "",y_axis:str = "",**kwargs):

    data = dist

    fig = sm.qqplot(data,line=line)

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()

def plot_dist_scatter(dist,title:str = "",x_axis:str = "",y_axis:str = "",**kwargs):
    data = dist

    fig = plt.scatter(data[:,0],data[:,1],**kwargs)

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()

def plot_scatter(x,y,title:str = "",x_axis:str = "",y_axis:str = "",**kwargs):

    fig = plt.scatter(x,y,**kwargs)

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()

def one_way_plot_restricted_model(dist,*,title:str = "Restricted",x_axis:str = "index",y_axis:str = "Y",linestyle = "--",**kwargs):
    #For this question pass the Dist object not Dist.dist


    mean = dist.cp_mean()
    plt.axhline(y=mean,xmin=0,xmax=dist.count,color="black")
    for i in range(dist.count):
        y_i = dist.dist[i]

        if y_i < mean:
            plt.vlines(i,ymin=y_i,ymax=mean,linestyle = linestyle, colors = "black")
        elif y_i > mean:
            plt.vlines(i,ymin=mean,ymax=y_i,linestyle = linestyle, colors="black")

    plt.scatter(range(dist.count),dist.dist,**kwargs)
    
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()

def one_way_plot_full_model(*dists,title:str = "Full",x_axis:str = "index",y_axis:str = "Y",linestyle = "--"):
    #For this question pass the Dist object not Dist.dist

    j = 1
    for dist in dists:
        mean = dist.cp_mean()
        plt.hlines(y=mean,xmin=j-1,xmax=dist.count+j-1,color="black")
        for i in range(dist.count):
            y_i = dist.dist[i]

            if y_i < mean:
                plt.vlines(i+j-1,ymin=y_i,ymax=mean,linestyle = linestyle, colors = "black")
            elif y_i > mean:
                plt.vlines(i+j-1,ymin=mean,ymax=y_i,linestyle = linestyle, colors="black")

        plt.scatter(range(j-1, dist.count + j - 1), dist.dist, label=dist.name, alpha=0.8)

        j += dist.count

    plt.legend()

    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    plt.show()