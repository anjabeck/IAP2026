"""
Usage:
python comparison.py
    --input path/to/sweights/in/hdf5/format.h5
    --a0 /path/to/a0/reference/file.root
    --a1 /path/to/a1/reference/file.root
    --aS /path/to/aS/reference/file.root
    --name test
    --qsq 1.1 7
    --mKpi 0.65 1.5
    --results path/to/results/file.yml
This produces plots called plots/test_mKpi_comparison.pdf and plots/test_q2_comparison.pdf
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str)
parser.add_argument('--a0',type=str)
parser.add_argument('--a1',type=str)
parser.add_argument('--aS',type=str)
parser.add_argument('--name',type=str,help="Output name")
parser.add_argument('--results',type=str,help="Results file")
parser.add_argument('--mKpi',type=float,nargs=2,default=[0.65,1.5])
parser.add_argument('--qsq',type=float,nargs=2,default=[2.0,11.0])
args = parser.parse_args()


import matplotlib.pyplot as plt
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

import hist

import pandas as pd
import numpy as np

labels = {
    'App' : r'$n_1^P$',
    'AS' : r'$n_0^S$',
    'A0' : r'$n_0^P$',
    'alpha' : r'$\alpha$',
    'beta' : r'$\beta$',
    'Nsig' : r'$N_{\rm sig}$',
    'Nbkg' : r'$N_{\rm bkg}$',
    'c' : r'$c$',
    'r' : r'$r$',
    'Aqc' : r'$n_{\beta}^c$',
    'Aqs' : r'$n_{\beta}^s$',
    'Aq' : r'$n_{\beta}$',
    'Cq' : r'$c_{\beta}$',
    'AfbHC' : r'$a_{hc}$',
    'AfbHS' : r'$a_{hs}$',
    'AfbLC' : r'$a_{\ell c}$',
    'AfbLS' : r'$a_{\ell s}$'
}

nbins = 100

import yaml

with open(args.results) as f:
    results = yaml.load(f, Loader=yaml.FullLoader)

if not ("Aqc" in results):
    results["Aqc"] = {"value": 0}
if not ("Aqs" in results):
    results["Aqs"] = {"value": 0}

data = pd.read_hdf(args.input)

import uproot
with uproot.open(args.a0) as f:
    # Check with Michele if this is the correct path
    dataA0 = f["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")
    dataA0["cosl"] = dataA0["cosThetaL"]
    dataA0["cosh"] = dataA0["cosThetaK"]
    dataA0 = dataA0.query(f"(mKpi>{args.mKpi[0]}) and (mKpi<{args.mKpi[1]})")
    dataA0 = dataA0.query(f"(q2>{args.qsq[0]}) and (q2<{args.qsq[1]})")
with uproot.open(args.a1) as f:
    dataA1 = f["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")
    dataA1["cosl"] = dataA1["cosThetaL"]
    dataA1["cosh"] = dataA1["cosThetaK"]
    dataA1 = dataA1.query(f"(mKpi>{args.mKpi[0]}) and (mKpi<{args.mKpi[1]})")
    dataA1 = dataA1.query(f"(q2>{args.qsq[0]}) and (q2<{args.qsq[1]})")
with uproot.open(args.aS) as f:
    dataAS = f["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")
    dataAS["cosl"] = dataAS["cosThetaL"]
    dataAS["cosh"] = dataAS["cosThetaK"]
    dataAS = dataAS.query(f"(mKpi>{args.mKpi[0]}) and (mKpi<{args.mKpi[1]})")
    dataAS = dataAS.query(f"(q2>{args.qsq[0]}) and (q2<{args.qsq[1]})")

print(len(data), len(dataA0)/len(data), len(dataA1)/len(data), len(dataAS)/len(data))
truth = {'wS': dataAS, 'wA0': dataA0, 'wApp': dataA1}
fs = (1-results["A0"]["value"]-results["App"]["value"]-results["Aqs"]["value"]-results["Aqc"]["value"]) * len(data) / len(dataAS)
# fs = (1-results["A0"]["value"]-results["App"]["value"]-results["Aq"]["value"]) * len(data) / len(dataAS)
# f0 = (1-0.0076930606805433385-0.36357486962416524-0.25534425308442404) * len(data) / len(dataA0)
# f1 = 0.25534425308442404 * len(data) / len(dataA1)
f1 = results["App"]["value"] * (len(data)/len(dataA1))
f0 = results["A0"]["value"] * (len(data)/len(dataA0))
# fs = ((1-0.0076930606805433385)*len(data) - (1-0.0076930606805433385-0.36357486962416524-0.25534425308442404) * len(dataA0) - 0.25534425308442404*len(dataA1)) / len(dataAS)
# f0 = ((1-0.0076930606805433385)*len(data) - fs * len(dataAS) - 0.25534425308442404*len(dataA1)) / len(dataA0)
# f1 = ((1-0.0076930606805433385)*len(data) - fs * len(dataAS) - f0 * len(dataA0)) / len(dataA1)
factor = {'wS': fs, 'wA0': f0, 'wApp': f1}

for key, label, unit in zip(['mKpi', 'q2'], [r'$m(K\pi)$', r'$q^2$'], [r'GeV/$c^2$', r'GeV$^2/c^4$']):
    mi, ma = data[key].min(), data[key].max()

    fig = plt.figure()
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 0.5, 0.5, 0.5], 'hspace':0.0}, sharex=True, figsize=[fig.get_size_inches()[0],fig.get_size_inches()[0]])
    lists = zip([1,2,3,4,4,4],[r"$n^S_0=\beta^2(|{A'}_0^L|^2+|{A'}_0^R|^2)$",r'$n_0^P=\beta^2(|{A}_0^L|^2+|{A}_0^R|^2)$',r'$n_1^P=\beta^2(|{A}_\perp^L|^2+|{A}_\perp^R|^2+|{A}_\parallel^L|^2+|{A}_\parallel^R|^2)$',r'$n_{\beta}$',r'$n_c^q$',r'$n_s^q$'],['wS','wA0','wApp','wAq','wAqc','wAqs'],['gold','navy','dodgerblue','firebrick','firebrick','firebrick'],["xx","//","\\\\","..","..",".."])
    # lists = zip([1,2,3,4,4,4],[r"$n^S_0$",r'$n_0^P$',r'$n_1^P$',r'$n_{\beta}$',r'$n_c^q$',r'$n_s^q$'],['wS','wA0','wApp','wAq','wAqc','wAqs'],['gold','navy','dodgerblue','firebrick','firebrick','firebrick'],["xx","//","\\\\","..","..",".."])
    maximum = 0
    for i,n,w,c,m in lists:
        print(w)
        if w not in data.columns:
            continue
        Hs = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
        Hs.fill(data[key], weight=data[w])
        mplhep.histplot(Hs, histtype="errorbar", label=n, xerr=True, yerr=True, color=c, ax=ax[0], marker='.')
        maximum = max(maximum, Hs.values().max())
        if i<4:
            Ht = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False))
            Ht.fill(truth[w][key])
            mplhep.histplot(factor[w]*Ht, histtype="step", color=c, ax=ax[0])
            residuals = (Hs.values()-factor[w]*Ht.values())/np.sqrt(Hs.variances()+factor[w]*Ht.values())
            for r in range(len(residuals)):
                color='black'
                ax[i].fill_between([Hs.axes[0].edges[r],Hs.axes[0].edges[r+1]], [0,0], [residuals[r],residuals[r]], color=c, linewidth=0)
            # ax[i].fill_between([mi,ma], [-3.0,-3.0], [3.0,3.0], color=c, alpha=0.3, linewidth=0, zorder=-10)
            ax[i].set_ylim(-3.5,3.5)
            # ax[i].set_ylabel(fr"Pull {n}",ha="right",y=1)
            ax[i].axhline(y=0, color='black', linewidth=1)
            ax[i].axhline(y=2, color='black', linestyle='dotted', linewidth=1)
            ax[i].axhline(y=-2, color='black', linestyle='dotted', linewidth=1)
            ax[i].set_yticks([-2,0,2])
            ax[i].set_yticklabels(["-2","0","2"])
    ax[3].set_xlabel(label+f" [{unit}]",ha="right",x=1)
    # ax[0].set_ylim(,)
    ax[0].axhline(y=0, color='black', linestyle='dotted', linewidth=1)
    ax[0].legend(handletextpad=0.1, fontsize=24)
    ax[0].set_ylabel("Value of the coefficient [a.u.]",ha="right",y=1)
    ax[0].set_xlim(mi,ma)
    ax[0].set_ylim(-maximum*0.05,maximum*1.1)
    ax[0].set_yticklabels([])
    plt.savefig(f"plots/{args.name}_{key}_comparison.pdf")
    plt.close()


    