import numpy as np
from scipy import stats
rng = np.random.default_rng(42)
cols = "Nstar Mstar central halo_index GroupID MHI BoverT Jstar_mag gasR80 dmR80 gas_kappa_rot star_kappa_rot angles Vx Vy Vz xpos ypos zpos Mstar_dic abs_mag_sdss_i app_mag_sdss_i abs_mag_irac1 app_mag_irac1 star_half_mass".split()
d = np.loadtxt('/mnt/user-data/uploads/m25n512_s50_galaxy_properties__1_.txt')
P = {c: d[:,i] for i,c in enumerate(cols)}
w = np.loadtxt('/mnt/user-data/uploads/W50.txt'); wid, W50 = w[:,0].astype(int), w[:,1]
gid = P['GroupID'].astype(int); lookup = dict(zip(wid,W50))
cut = (P['Mstar_dic']>1e8)&(P['MHI']>1e8)
print("Total galaxies:", len(d))
print("With W50:", np.isin(gid,wid).sum())
print("Pass mass cuts (all):", cut.sum())
print("Mstar_dic>1e8 only:", (P['Mstar_dic']>1e8).sum(), " MHI>1e8 only:", (P['MHI']>1e8).sum())
sel = cut & np.isin(gid,wid); print("Final matched:", sel.sum())
print("W50 galaxies failing cuts:", np.isin(gid,wid).sum()-sel.sum())
M = P['abs_mag_irac1'][sel]; W = np.array([lookup[g] for g in gid[sel]]); x = np.log10(W)
cen = P['central'][sel]==1
print("\nIRAC1 abs mag (all galaxies): min %.2f max %.2f median %.2f"%(np.nanmin(P['abs_mag_irac1']),np.nanmax(P['abs_mag_irac1']),np.nanmedian(P['abs_mag_irac1'])))
print("IRAC1 abs mag (matched): min %.2f max %.2f median %.2f std %.2f"%(M.min(),M.max(),np.median(M),M.std()))
print("W50 (matched) km/s: min %.1f max %.1f median %.1f ; logW min %.2f max %.2f mean %.3f std %.3f"%(W.min(),W.max(),np.median(W),x.min(),x.max(),x.mean(),x.std()))
print("Mstar_dic matched median %.2e ; MHI median %.2e"%(np.median(P['Mstar_dic'][sel]),np.median(P['MHI'][sel])))
print("Centrals %d (%.1f%%), satellites %d"%(cen.sum(),100*cen.mean(),(~cen).sum()))
r,pv = stats.pearsonr(x,M); rho,ps = stats.spearmanr(x,M)
print("Pearson r %.3f (p=%.1e) ; Spearman rho %.3f (p=%.1e)"%(r,pv,rho,ps))

def ols(x,y):
    b=np.cov(x,y,bias=True)[0,1]/np.var(x); return b
def allfits(x,y):
    b1=ols(x,y); b2=1/ols(y,x)
    sxx,syy,sxy=np.var(x),np.var(y),np.cov(x,y,bias=True)[0,1]
    bis=(b1*b2-1+np.sqrt((1+b1**2)*(1+b2**2)))/(b1+b2)
    orth=(syy-sxx+np.sqrt((syy-sxx)**2+4*sxy**2))/(2*sxy)
    return {'direct':b1,'inverse':b2,'bisector':bis,'orthogonal':orth}
def table(x,y,label,nb=2000):
    f=allfits(x,y); bs={k:[] for k in f}
    for _ in range(nb):
        i=rng.integers(0,len(x),len(x)); r=allfits(x[i],y[i])
        for k in r: bs[k].append(r[k])
    print(f"\n{label}  (N={len(x)})")
    print(f"{'fit':11s} {'slope':>7s} {'err':>6s} {'16-84%':>16s} {'p':>5s} {'M@logW=2.0':>11s} {'rms(M)':>7s}")
    for k in f:
        b=f[k]; a=y.mean()-b*x.mean(); res=y-(a+b*x)
        lo,hi=np.percentile(bs[k],[16,84])
        print(f"{k:11s} {b:7.2f} {np.std(bs[k]):6.2f} [{lo:6.2f},{hi:6.2f}] {abs(b)/2.5:5.2f} {a+b*2.0:11.2f} {res.std():7.2f}")
    return f
table(x,M,"All matched")
table(x[cen],M[cen],"Centrals only")
table(x[~cen],M[~cen],"Satellites only")

# residuals vs logW about inverse fit (curvature check): quadratic term
b=allfits(x,M)['inverse']; a=M.mean()-b*x.mean(); res=M-(a+b*x)
c=np.polyfit(x,M,2); print("\nQuadratic fit M = c2 x^2 + c1 x + c0:", c)
# inverse quadratic: fit x = quadratic in M (inverse approach), check curvature significance by bootstrap
cs=[]
for _ in range(2000):
    i=rng.integers(0,len(x),len(x)); cs.append(np.polyfit(M[i],x[i],2)[0])
cs=np.array(cs); print("Inverse quadratic coeff (logW vs M) c2: %.5f +/- %.5f"%(np.polyfit(M,x,2)[0],cs.std()))
# binned medians
edges=np.percentile(x,np.linspace(0,100,7)); print("\nBinned in logW (6 equal-count bins): logW_med, M_med, M_scatter(std), N")
bx,by=[],[]
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(x>=lo)&(x<=hi) if hi==edges[-1] else (x>=lo)&(x<hi)
    bx.append(np.median(x[m])); by.append(np.median(M[m])); print("%.2f  %.2f  %.2f  %d"%(np.median(x[m]),np.median(M[m]),M[m].std(),m.sum()))
print("Binned-median slope (OLS on medians): %.2f"%np.polyfit(bx,by,1)[0])
# low-W check (<90 km/s -> logW<1.954)
lo=x<np.log10(90); print("\nN with W50<90 km/s: %d (%.1f%%)"%(lo.sum(),100*lo.mean()))
b_hi=allfits(x[~lo],M[~lo])['inverse']; b_lo=allfits(x[lo],M[lo])['inverse']
print("Inverse slope W50>=90: %.2f ; W50<90: %.2f"%(b_hi,b_lo))
