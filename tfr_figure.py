# Makes the final Tully-Fisher figure. Needs tfr_matched_sample.csv (written by analysis.py) in /mnt/user-data/outputs/
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = np.loadtxt('/mnt/user-data/outputs/tfr_matched_sample.csv', delimiter=',')
x, M = d[:,2], d[:,3]
def ols(a,b): return np.cov(a,b,bias=True)[0,1]/np.var(a)
b1 = ols(x,M); b2 = 1/ols(M,x)
bis = (b1*b2-1+np.sqrt((1+b1**2)*(1+b2**2)))/(b1+b2)
sxx,syy,sxy = np.var(x),np.var(M),np.cov(x,M,bias=True)[0,1]
orth = (syy-sxx+np.sqrt((syy-sxx)**2+4*sxy**2))/(2*sxy)
fits = [('Inverse OLS (headline)', b2, '#C44E52','-'), ('Orthogonal', orth, '#8172B3',':'), ('Bisector', bis, '#4C72B0','-.'), ('Direct OLS', b1, '#55A868','--')]
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(x, M, s=14, c='k', alpha=.65, lw=0, label=f'SIMBA galaxies (N={len(x)})')
xx = np.linspace(1.4, 2.8, 50)
for lab,b,c,ls in fits:
    a = M.mean()-b*x.mean()
    ax.plot(xx, a+b*xx, color=c, ls=ls, lw=2.2, label=f'{lab}: {b:.2f} mag/dex')
ax.set_xlim(1.35, 2.85); ax.set_ylim(-14, -24)
ax.set_xlabel(r'$\log_{10}(W_{50}\ [\mathrm{km\,s^{-1}}])$')
ax.set_ylabel('IRAC1 (3.6 μm) absolute magnitude [mag]')
ax.legend(frameon=False, fontsize=9, loc='upper left')
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/tfr_report_figure.pdf'); plt.savefig('/mnt/user-data/outputs/tfr_report_figure.png', dpi=200)
