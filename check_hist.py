# Checks the placeholder (100.0) IRAC1 magnitudes, horizontal scatter, and regenerates the histogram.
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cols = "Nstar Mstar central halo_index GroupID MHI BoverT Jstar_mag gasR80 dmR80 gas_kappa_rot star_kappa_rot angles Vx Vy Vz xpos ypos zpos Mstar_dic abs_mag_sdss_i app_mag_sdss_i abs_mag_irac1 app_mag_irac1 star_half_mass".split()
d = np.loadtxt('/mnt/user-data/uploads/m25n512_s50_galaxy_properties__1_.txt')
P = {c: d[:,i] for i,c in enumerate(cols)}
am = P['abs_mag_irac1']
bad = am>=50
print("abs_mag_irac1 >= 50:", bad.sum(), "| unique bad values:", np.unique(am[bad]))
print("their Nstar:", np.unique(P['Nstar'][bad])[:5], " Mstar_dic max among bad: %.2e"%P['Mstar_dic'][bad].max())
print("In cut sample:", (bad&(P['Mstar_dic']>1e8)&(P['MHI']>1e8)).sum())
good = am[~bad]; print("valid N:", len(good), "range %.2f to %.2f, median %.2f"%(good.min(),good.max(),np.median(good)))

# horizontal scatter about inverse fit
w = np.loadtxt('/mnt/user-data/uploads/W50.txt'); lookup=dict(zip(w[:,0].astype(int),w[:,1]))
gid=P['GroupID'].astype(int); cut=(P['Mstar_dic']>1e8)&(P['MHI']>1e8)&np.isin(gid,w[:,0].astype(int))
M=am[cut]; x=np.log10([lookup[g] for g in gid[cut]])
d_,c_=np.polyfit(M,x,1)   # logW = d*M + c
print("Inverse fit logW = %.4f*M + %.3f ; rms in logW = %.3f dex (= %.1f%% in W)"%(d_,c_,np.std(x-(d_*M+c_)),100*(10**np.std(x-(d_*M+c_))-1)))
print("equivalent rms in M along vertical for that slope: %.2f; in M = rms_logW/|d| = %.2f"%(np.std(M-(M.mean()+(x-x.mean())/d_)), np.std(x-(d_*M+c_))/abs(d_)))

# corrected histogram
fig, ax = plt.subplots(figsize=(7,4.5))
ax.hist(good, bins=40, color='#4C72B0', alpha=.5, label=f'All galaxies with valid IRAC1 (N={len(good)})')
ax.hist(M, bins=40, color='#DD8452', alpha=.8, label=f'Matched, mass cuts (N={len(M)})')
ax.set_xlabel('IRAC1 absolute magnitude [mag]'); ax.set_ylabel('Number of galaxies'); ax.invert_xaxis(); ax.legend(frameon=False)
for s in ['top','right']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('hist_irac1.pdf'); plt.savefig('hist_irac1.png', dpi=150)
