If you havn't used Python before you may find these hints useful!

## Question 2

In the notebook provided we compute one reailization of the process Z. You then need to simulate the process M times and do some statisitics.

- Running the simulations. To do this I recommend you start by initializing an array
```
Z_all=np.zeros((M,steps))                #steps is a variable already defined=101
```
Then fill up the rows one at a time.
```
for i in range(M):
    # run a simulation to get Z
    Z_all[i,:]=Z
```    
- Mean and standard deviation

To calculate mean and standard deviation use the function np.mean() and np.std(). The optional argument 'axis' tells the function which axis of the array to take the mean over.
```
means=np.mean(Z_all,axis=0)             #to check shape use np.shape(means)
stdvs=np.std(Z_all,axis=0)
```
- Plotting

To make our plot use plt.errorbar().
```
plt.errorbar(range(steps),means,yerr=stdvs,fmt='o',mew=2)
plt.savefig('myplot.pdf')
```
On the x-axis we are plotting 'range(steps)'. On the y-axis we plot the means plus/minus the stdvs (argument 'yerr'). The optional arguments 'fmt' and 'mew' are specifying the marker format and size.

- Empirical PDF

To compute the empirical PDF I recommend you use gaussian_kde.
```
from scipy import stats # you need this package

timestep=20
data=Z_all[:,timestep]

empiricalPDF_func=stats.gaussian_kde(data,bw_method=0.05)      # returrns a function

x=np.linspace(data.min(),data.max(),1000)
plt.plot(x,empiricalPDF_func(x))
```
gaussian_kde takes an optional parameter 'bw_method' which is basically the bandwidth. You may need to play with this. gaussian_kde returns a function.

- Compare with analytic PDF

We need to compare the empirical PDF with the analytic PDF (derived in the sheet using $f_Z(z)dz=f_Y(y)dy$ where $Z:=\exp{Y}$). To do this you may like to define your own funtion. E.g.
```
def lognorm(x,mu,sigma,n):
    y=1/x*1/np.sqrt(2*np.pi*n*sigma**2)*np.exp(-(np.log(x)-n*mu)**2/(2*n*sigma**2))
    return y
```
Then do something like
```
timestep=20
data=Z_all[:,timestep]

x=np.linspace(data.min(),data.max(),1000)
plt.plot(x,lognorm(x,0.0,0.2,timestep))
```
For plotting both curves on the same plot and making a legend see the quickstart notebook
