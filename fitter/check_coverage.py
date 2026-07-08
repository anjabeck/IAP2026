"""
python check_coverage.py --input path/to/toy/sweights/*.h5 --name output/prefix --data path/to/data.h5
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,help="Files",nargs="+")
parser.add_argument('--name',type=str,help="Output name")
parser.add_argument('--data',type=str,help="Data file")
parser.add_argument('--mKpi',type=float,nargs=2,default=[0.65,1.5])
args = parser.parse_args()


import matplotlib.pyplot as plt
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

import hist

import pandas as pd
import numpy as np

from myconstants import *

nbins = 15

dataall = pd.read_hdf(args.data)

mih, mah = dataall["mKpi"].min(), dataall["mKpi"].max()
miq, maq = dataall["q2"].min(), dataall["q2"].max()

for t in ['total', 'wApp', 'wA0', 'wS', 'wAqc', 'wAqs', 'wAq']:
    pulls = []
    if t=='total':
        Href = hist.Hist(hist.axis.Regular(nbins, mih, mah, underflow=False, overflow=False), hist.axis.Regular(nbins, miq, maq, underflow=False, overflow=False), storage=hist.storage.Weight())
        Href.fill(dataall["mKpi"], dataall["q2"], weight=45000/len(dataall))
    else:
        if t in dataall.keys():
            Href = hist.Hist(hist.axis.Regular(nbins, mih, mah, underflow=False, overflow=False), hist.axis.Regular(nbins, miq, maq, underflow=False, overflow=False), storage=hist.storage.Weight())
            Href.fill(dataall["mKpi"], dataall["q2"], weight=dataall[t] * 45000/len(dataall))
    for i,f in enumerate(args.input):
        Hpulls = hist.Hist(hist.axis.Regular(50, -5,5 , underflow=False, overflow=False))
        print("Processing",i,"/",len(args.input))
        try:
            data = pd.read_hdf(f)
        except:
            continue
        if t=='total':
            Hi = hist.Hist(hist.axis.Regular(nbins, mih, mah, underflow=False, overflow=False), hist.axis.Regular(nbins, miq, maq, underflow=False, overflow=False))
            Hi.fill(data["mKpi"], data["q2"])
        else:
            if t in data.keys():
                Hi = hist.Hist(hist.axis.Regular(nbins, mih, mah, underflow=False, overflow=False), hist.axis.Regular(nbins, miq, maq, underflow=False, overflow=False), storage=hist.storage.Weight())
                Hi.fill(data["mKpi"], data["q2"], weight=data[t])
        pull = (Hi.values() - Href.values()) / np.sqrt(Hi.variances())
        pulls.append(pull)
        for p in pull:
            Hpulls.fill(p)
        mplhep.histplot(Hpulls, histtype="errorbar", xerr=True, yerr=True)
    plt.xlabel(f"Pull {t}",ha="right",x=1)
    plt.ylabel("Entries",ha="right",y=1)
    plt.savefig(f"plots/{args.name}_{t}_pulls.pdf")
    plt.close()
    pulls = np.array(pulls)
    mean_per_bin = np.nanmean(pulls, axis=0)
    std_per_bin = np.nanstd(pulls, axis=0)
    mean_per_bin = np.nanmean(pulls, axis=0)
    std_per_bin = np.nanstd(pulls, axis=0)
    # A[~np.isnan(A)].mean()
    plt.imshow(mean_per_bin, cmap="coolwarm", origin="lower", extent=[miq, maq, mih, mah], aspect="auto", vmin=-3, vmax=3)
    plt.xlabel(r"$q^2$ [GeV$^2/c^4$]",ha="right",x=1)
    plt.ylabel(r"$m(K\pi)$ [GeV/$c^2$]",ha="right",y=1)
    plt.colorbar()
    plt.savefig(f"plots/{args.name}_{t}_mean.pdf")
    plt.close()
    plt.imshow(std_per_bin, cmap="coolwarm", origin="lower", extent=[miq, maq, mih, mah], aspect="auto", vmin=0, vmax=2)
    plt.xlabel(r"$q^2$ [GeV$^2/c^4$]",ha="right",x=1)
    plt.ylabel(r"$m(K\pi)$ [GeV/$c^2$]",ha="right",y=1)
    plt.colorbar()
    plt.savefig(f"plots/{args.name}_{t}_std.pdf")
    plt.close()
    plt.imshow(Href.values(), origin="lower", extent=[miq, maq, mih, mah], aspect="auto")
    plt.xlabel(r"$q^2$ [GeV$^2/c^4$]",ha="right",x=1)
    plt.ylabel(r"$m(K\pi)$ [GeV/$c^2$]",ha="right",y=1)
    plt.colorbar()
    plt.savefig(f"plots/{args.name}_{t}_ref.pdf")
    plt.close()