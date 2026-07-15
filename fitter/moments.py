import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,help="All sweights from the toys",nargs="+")
parser.add_argument('--name',type=str,help="Output name")
parser.add_argument('--data',type=str,help="Data file",nargs="+")
parser.add_argument('--mKpi',type=float,nargs=2,default=[0.65,1.5])
parser.add_argument('--qsq',type=float,nargs=2,default=[2,10])
parser.add_argument('--binnedfits',type=str,help="Binned fits (we don't need this for now)",nargs="+",default=[])
parser.add_argument('--center', action=argparse.BooleanOptionalAction, help="Center around true value? Default: --center means true and --no-center means false")
parser.add_argument('--nbins',type=int,default=5)
args = parser.parse_args()


import matplotlib.pyplot as plt
import mplhep
mplhep.style.use(mplhep.style.LHCb2)

import hist

import pandas as pd
import numpy as np


import scipy
leg0 = scipy.special.legendre(0)
leg1 = scipy.special.legendre(1)
leg2 = scipy.special.legendre(2)

def S2s(cosh, cosl):
    truth = (5/6) * leg2(cosl)*(2*leg0(cosh) - 5*leg2(cosh))
    scaling = 4
    return scaling * truth

def S2c(cosh, cosl):
    truth = (5/3) * leg2(cosl)*(leg0(cosh) + 5*leg2(cosh))
    scaling = -1
    return scaling * truth

def MomUnc(funcvals):
    Norm =len(funcvals)
    K = np.sum(funcvals) / Norm
    arg = (funcvals - K)
    Unc = np.sqrt( np.sum( arg*arg ) ) / Norm
    return K, Unc

def Covariance(func1, func2):
    Norm = len(func1)
    K1 = np.sum(func1) / Norm
    K2 = np.sum(func2) / Norm
    arg1 = (func1 - K1)
    arg2 = (func2 - K2)
    Cov = np.sum( arg1 * arg2 ) / Norm
    return Cov

from myconstants import *

if len(args.binnedfits)>0:
    results = {
        "A0": {"values": [], "unc_lo": [], "unc_hi": []},
        "App": {"values": [], "unc_lo": [], "unc_hi": []},
        "qsq": [i+0.125 for i in range(1,len(args.binnedfits)+1)],
        "qsq": np.linspace(1,7,len(args.binnedfits)),
        "qerr": [0.125 for i in range(1,len(args.binnedfits)+1)],
    }
    results["qsq"][0] = 1.1 + (1.25-1.1)/2
    results["qerr"][0] = 0.075

def errordiv(a,b,da,db,cov):
    return np.abs(a/b) * np.sqrt((da/a)**2 + (db/b)**2 - 2*cov/(a*b))

for bf in args.binnedfits:
    print(bf)
    with open(bf) as f:
        res = yaml.load(f, Loader=yaml.FullLoader)
        cov_A0_App = res["A0"]["covariance"]["App"]

        Norm = res["A0"]["value"]+res["App"]["value"]
        Norm_hi = np.sqrt(res["A0"]["error_upper"]**2 + res["App"]["error_upper"]**2 + 2*cov_A0_App)
        Norm_lo = np.sqrt(res["A0"]["error_lower"]**2 + res["App"]["error_lower"]**2 + 2*cov_A0_App)

        results["A0"]["values"].append(res["A0"]["value"]/Norm)
        results["App"]["values"].append(res["App"]["value"]/Norm)

        # propagation
        results["A0"]["unc_lo"].append(errordiv(res["A0"]["value"],Norm,res["A0"]["error_lower"],Norm_lo,res["A0"]["error_lower"]**2+cov_A0_App))
        results["A0"]["unc_hi"].append(errordiv(res["A0"]["value"],Norm,res["A0"]["error_upper"],Norm_hi,res["A0"]["error_upper"]**2+cov_A0_App))
        results["App"]["unc_lo"].append(errordiv(res["App"]["value"],Norm,res["App"]["error_lower"],Norm_lo,res["App"]["error_lower"]**2+cov_A0_App))
        results["App"]["unc_hi"].append(errordiv(res["App"]["value"],Norm,res["App"]["error_upper"],Norm_hi,res["App"]["error_upper"]**2+cov_A0_App))
        print("App", res["App"]["value"], res["App"]["error_lower"], res["App"]["error_upper"])
        print("A0", res["A0"]["value"], res["A0"]["error_lower"], res["A0"]["error_upper"])


if len(args.binnedfits)>0:
    print(results)
    nbins = len(args.binnedfits)
else:
    nbins = args.nbins

import uproot
first = True
for d in args.data:
    with uproot.open(d) as f:
        di = f["B02KstMuMu_Run1_centralQ2E_sig"].arrays(library="pd")
        di["cosl"] = di["cosThetaL"]
        di["cosh"] = di["cosThetaK"]
        di = di.query(f"mKpi>{args.mKpi[0]} and mKpi<{args.mKpi[1]}")
        di = di.query(f"q2>{args.qsq[0]} and q2<{args.qsq[1]}")
    if first:
        dataall = di
        first = False
    else:
        dataall = pd.concat([dataall,di])
    datasmall = dataall.sample(n=45000)


def get_corr(data,H,weights):
    covall = np.zeros((len(H.values()),len(H.values())))
    for b1 in range(len(H.values())):
        xmi, xma = H.axes[0].edges[b1], H.axes[0].edges[b1+1]
        seli = (data.q2.values>np.ones(len(data))*xmi) & (data.q2.values<np.ones(len(data))*xma)
        for b2 in range(b1+1):
            ymi, yma = H.axes[0].edges[b2], H.axes[0].edges[b2+1]
            selj = (data.q2.values>np.ones(len(data))*ymi) & (data.q2.values<np.ones(len(data))*yma)
            covall[b1,b2] = Covariance(weights[seli], weights[selj])
    # get correlation
    corrall = np.zeros((len(H.values()),len(H.values())))
    for b1 in range(len(H.values())):
        for b2 in range(b1+1):
            corrall[b1,b2] = covall[b1,b2] / np.sqrt(covall[b1,b1]*covall[b2,b2])
            corrall[b2,b1] = corrall[b1,b2]
    return corrall


ranges = np.linspace(1,7,nbins+1)
ranges[0] = 1.1
for key, label, unit in zip(['q2'], [r'$q^2$'], [r'GeV$^2/c^4$']):
    for (mi,ma),nbins in zip([(ranges[0],ranges[1]),(ranges[1],ranges[-1])],[1,nbins-1]):
        hists = {'wS': pd.DataFrame(), 'wApp': pd.DataFrame(), 'wA0': pd.DataFrame()}
        errs = {'wS': pd.DataFrame(), 'wApp': pd.DataFrame(), 'wA0': pd.DataFrame()}

        for i,f in enumerate(args.input):
            print("Processing",i,"/",len(args.input))
            try:
                data = pd.read_hdf(f)
            except Exception as e:
                print(e)
                continue
            Hall = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
            Hall.fill(data[key], weight=data['wApp']+data['wA0'])
            corrall = get_corr(data,Hall,data['wApp']+data['wA0'])
            # get covariance matrix
            covall = np.zeros((nbins,nbins))
            for b1 in range(nbins):
                for b2 in range(b1+1):
                    covall[b1,b2] = corrall[b1,b2] * np.sqrt(Hall.variances()[b1] * Hall.variances()[b2])
                    covall[b2,b1] = covall[b1,b2]
            # get corrected uncertainty
            errall = np.sqrt(np.sum(covall,axis=1))
            for w in ['wApp','wA0']:
                H = hist.Hist(hist.axis.Regular(nbins, mi, ma, underflow=False, overflow=False), storage=hist.storage.Weight())
                if not w in data.keys():
                    print(data)
                    continue
                H.fill(data[key], weight=data[w])
                corrw = get_corr(data,H,data[w])
                covw = np.zeros((nbins,nbins))
                for b1 in range(nbins):
                    for b2 in range(b1+1):
                        covw[b1,b2] = corrw[b1,b2] * np.sqrt(H.variances()[b1] * H.variances()[b2])
                        covw[b2,b1] = covw[b1,b2]
                errw = np.sqrt(np.sum(covw,axis=1))
                hists[w][i] = H.values()/Hall.values()
                # Error propagation
                covs = np.zeros(nbins)
                for b in range(nbins):
                    xmi, xma = Hall.axes[0].edges[b], Hall.axes[0].edges[b+1]
                    datai = data.query(f"{key}>{xmi} and {key}<{xma}")
                    correlation = Covariance(datai[w], datai['wApp']+datai['wA0']) / np.sqrt(Covariance(datai[w], datai[w]) * Covariance(datai['wApp']+datai['wA0'], datai['wApp']+datai['wA0']))
                    covs[b] = correlation * np.sqrt(errw[b]**2 * errall[b]**2)
                errs[w][i] = hists[w][i] * np.sqrt(errall**2/Hall.values()**2 + errw**2/H.values()**2 - 2*covs/(Hall.values()*H.values()))

        x = Hall.axes[0].centers
        xerr = Hall.axes[0].widths/2
        ySum = np.zeros(nbins)
        mini, maxi = 0, 1
        for n,w,cf,a,h in zip([r'$-S_{2c}$',r'$4S_{2s}$'],['wA0','wApp'],['navy','dodgerblue'],[0.50,0.50],['//','\\\\']):
            print(len(hists[w]))
            if len(hists[w])==0:
                continue

            y = hists[w][0]
            mini, maxi = min(mini, y.min()), max(maxi, y.max())
            yerr = np.where(errs[w].median(axis=1)>0, errs[w].median(axis=1), 1e-5)
            xerr = Hall.axes[0].widths/2
            means = []
            for i in range(nbins):
                xmi, xma = x[i]-xerr[i], x[i]+xerr[i]
                seliall = dataall.query(f"{key}>{xmi} and {key}<{xma}")
                selismall = datasmall.query(f"{key}>{xmi} and {key}<{xma}")
                if w=='wA0':
                    momentall = S2c(seliall["cosh"],seliall["cosl"])
                    momentsmall = S2c(selismall["cosh"],selismall["cosl"])
                elif w=='wApp':
                    momentall = S2s(seliall["cosh"],seliall["cosl"])
                    momentsmall = S2s(selismall["cosh"],selismall["cosl"])
                else:
                    continue
                Ktruth,Utruth = MomUnc(momentall)[0], MomUnc(momentall)[1]
                if args.center:
                    K = Ktruth
                    U = MomUnc(momentsmall)[1]
                else:
                    K,U = MomUnc(momentsmall)[0], MomUnc(momentsmall)[1]
                plt.fill_between([xmi,xma],[K-U]*2,[K+U]*2,edgecolor="w",linewidth=0,facecolor=cf, alpha=0.6,hatch=h)

                means.append(K)
            if args.center:
                plt.errorbar(x,means,yerr=yerr,xerr=xerr,color="k", ecolor="k", fmt='none', elinewidth=1, capthick=1, zorder=10)#markeredgecolor='w', 
            else:
                plt.errorbar(x,y,yerr=yerr,xerr=xerr,color="k", ecolor="k", fmt='.', elinewidth=1, capthick=1, zorder=10)#markeredgecolor='w', 
            if nbins==1:
                plt.errorbar(0,0,yerr=0,xerr=0,color="k", ecolor="k", fmt='.', elinewidth=1, capthick=1, zorder=10,label=f"{n} (sPlot)")
                plt.fill_between([mi,ma],[0,0],[0,0],edgecolor="w",linewidth=0,facecolor=cf, alpha=0.6,hatch=h, label=f"{n} (Moment)")
                if len(args.binnedfits)>0:
                    if args.center:
                        plt.errorbar(results["qsq"][0], means, xerr=results["qerr"][0], yerr=[results[w[1:]]["unc_lo"][0], results[w[1:]]["unc_hi"]][0], elinewidth=3, capthick=1, fmt='none', color="w", ecolor="w", label=f"{n} (Fit)")
                    else:
                        plt.errorbar(results["qsq"][0], results[w[1:]]['values'][0], xerr=results["qerr"][0], yerr=[results[w[1:]]["unc_lo"][0], results[w[1:]]["unc_hi"]][0], elinewidth=3, capthick=1, fmt='.', color="gray", ecolor="gray", label=f"{n} (Fit)")
            else:
                if len(args.binnedfits)>1:
                    if args.center:
                        plt.errorbar(results["qsq"][1:], means, xerr=results["qerr"][1:], yerr=[results[w[1:]]["unc_lo"][1:], results[w[1:]]["unc_hi"][1:]], elinewidth=3, capthick=1, fmt='none', color="w", ecolor="w")
                    else:
                        plt.errorbar(results["qsq"][1:], results[w[1:]]['values'][1:], xerr=results["qerr"][1:], yerr=[results[w[1:]]["unc_lo"][1:], results[w[1:]]["unc_hi"][1:]], elinewidth=3, capthick=1, fmt='.', color="gray", ecolor="gray")
    plt.ylim(0.05,1.1)
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(args.binnedfits)>0:
        order = [0,1,3,5,2,4]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], handlelength=1.2, facecolor='gray', framealpha=0.3, frameon=True, ncols=3, loc="upper center", columnspacing=0.8)
        plt.ylim(0.05,1.1)
    else:
        order = [0,1,2,3]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], handlelength=1.2, facecolor='gray', framealpha=0.3, frameon=True, ncols=2, loc="upper right", columnspacing=0.8)
        plt.ylim(-0.2,1.3)
    plt.xlim(args.qsq[0],args.qsq[1])
    plt.xlabel(label+f" [{unit}]",ha="right",x=1)
    plt.ylabel(fr"Value of the coefficient",ha="right",y=1)
    if args.center:
        plt.savefig(f"plots/{args.name}_{key}_moments_center.pdf")
    else:
        plt.savefig(f"plots/{args.name}_{key}_moments_scatter.pdf")
    plt.close()
