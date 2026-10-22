import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(42) # random seed for the bootstrap

cols = "Nstar Mstar central halo_index GroupID MHI BoverT Jstar_mag gasR80 dmR80 gas_kappa_rot star_kappa_rot angles Vx Vy Vz xpos ypos zpos Mstar_dic abs_mag_sdss_i app_mag_sdss_i abs_mag_irac1 app_mag_irac1 star_half_mass".split()
d = np.loadtxt("C:\\Users\\Slime\\Downloads\\m25n512_s50_galaxy_properties.txt")    # Load data by retrive data
P = {c: d[:,i] for i,c in enumerate(cols)}
print("GroupID == row index for all rows:", np.all(P['GroupID'].astype(int)==np.arange(len(d))))
w = np.loadtxt("C:\\Users\\Slime\\Downloads\\W50.txt") # W50.txt

wid, W50 = w[:,0].astype(int), w[:,1]

cut = (P['Mstar_dic']>1e8)&(P['MHI']>1e8)
gid = P['GroupID'].astype(int)
lookup = dict(zip(wid, W50))
sel = cut & np.isin(gid, wid)
M = P['abs_mag_irac1'][sel]
x = np.log10(np.array([lookup[g] for g in gid[sel]]))
print("N matched after cuts:", sel.sum(), "| finite:", np.isfinite(M).sum(), np.isfinite(x).sum())
print("M range", M.min(), M.max(), " logW range", x.min(), x.max())
print("central frac", P['central'][sel].mean())

# Histogram of abs_mag_irac1 (all galaxies, and those with cuts)
fig, ax = plt.subplots(figsize=(7,4.5))

am = P['abs_mag_irac1']
am = am[np.isfinite(am) & (am < 50)]   # exclude placeholder value 100.0 (60 galaxies)
ax.hist(am, bins=40, color='#4C72B0', alpha=.5, label=f'All galaxies with valid IRAC1 (N={len(am)})')
ax.hist(M, bins=40, color='#DD8452', alpha=.8, label=f'Matched, cuts (N={len(M)})')
ax.set_xlabel('abs_mag_irac1 [mag]'); ax.set_ylabel('Number of galaxies'); ax.legend()
ax.invert_xaxis()
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('hist_irac1.pdf'); plt.savefig('hist_irac1.png', dpi=150); plt.close()

# Fitting
def ols_yx(x,y):
    b = np.cov(x,y,bias=True)[0,1]/np.var(x); return b, y.mean()-b*x.mean()
def fits(x,y):
    out = {}
    b1,_ = ols_yx(x,y)                      # M on logW
    b2i,_ = ols_yx(y,x); b2 = 1/b2i         # inverse: logW on M, expressed as dM/dlogW
    sxx,syy,sxy = np.var(x),np.var(y),np.cov(x,y,bias=True)[0,1]
    bis = (b1*b2-1+np.sqrt((1+b1**2)*(1+b2**2)))/(b1+b2)
    orth = (syy-sxx+np.sqrt((syy-sxx)**2+4*sxy**2))/(2*sxy)
    out['OLS(M|logW)']=b1; out['OLS inverse (logW|M)']=b2; out['Bisector']=bis; out['Orthogonal']=orth
    return out
res = fits(x,M)
boot = {k:[] for k in res}
n=len(x)
for _ in range(2000):
    i = rng.integers(0,n,n); r = fits(x[i],M[i])
    for k in r: boot[k].append(r[k])
print("\nSlope dM/dlog10(W50) [mag/dex]  (index p in L ~ W^p is |slope|/2.5)")
for k in res:
    s = np.std(boot[k]); print(f"{k:24s} {res[k]:8.3f} +/- {s:.3f}   p = {abs(res[k])/2.5:5.2f}")
print("Pearson r:", np.corrcoef(x,M)[0,1])
print("Scatter in M about bisector fit (mag):", np.std(M-(res['Bisector']*x+(M.mean()-res['Bisector']*x.mean()))))

# TFR plot
fig, ax = plt.subplots(figsize=(7.5,5.5))
ax.scatter(x, M, s=14, c='k', alpha=.7, lw=0)
xx = np.linspace(1.3,2.9,50)
styles = {'OLS(M|logW)':('#C44E52','-'),'OLS inverse (logW|M)':('#55A868','--'),'Bisector':('#4C72B0','-.')}
for k,(c,ls) in styles.items():
    b=res[k]; a=M.mean()-b*x.mean()
    ax.plot(xx, a+b*xx, c=c, ls=ls, lw=2, label=f'{k}: slope {b:.2f}')
ax.invert_yaxis()
ax.set_xlabel(r'$\log_{10}(W_{50}\ [\mathrm{km\,s^{-1}}])$'); ax.set_ylabel('IRAC1 (3.6 μm) absolute magnitude')
ax.legend(frameon=False, fontsize=9)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('tfr_fits.pdf'); plt.savefig('tfr_fits.png', dpi=150); plt.close()
np.savetxt('tfr_matched_sample.csv', np.c_[gid[sel], 10**x, x, M], delimiter=',', header='GroupID,W50_kms,logW50,abs_mag_irac1', fmt=['%d','%.4f','%.5f','%.4f'])
