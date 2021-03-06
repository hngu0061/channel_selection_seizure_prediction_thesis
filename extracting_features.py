import scipy.io as sio
import numpy as np
import glob
import os
import scipy.misc as misc
from sklearn import preprocessing
import scipy as sp
import scipy.signal as spsig
import pandas as pd
import scipy.stats as spstat
import json
import time

# Function to load data from Matlab files
def get_data(file):
    matfile = sio.loadmat(file)
    data = (matfile["data"]).T
    return data


# Function to extract short-term features (Arete's study)
def short_features(pat, outfile, datapath, channelSet):
    f = datapath + "/*mat"
    ff = glob.glob(f)

    label = [str(os.path.basename(n)) for n in ff]
    channel_str = ""
    for channel in channelSet:
        channel_str += "{}-".format(channel + 1)
    channel_str = channel_str.rstrip("-")
    output = []
    featureList = []
    mydata = []
    bands = [0.1, 4, 8, 12, 30, 70, 180]
    rate = 400.0
    for i in range(len(ff)):
        data_full = get_data(ff[i])
        output = []
        featureList = []
        featureList.append("File")
        output.append(label[i])
        featureList.append("pat")
        output.append(pat)

        for j in range(19):
            if os.path.basename(ff[i]) == "1_45_1.mat":
                continue
            data = data_full[
                :, j * int(rate * 60 / 2) : (j) * int(rate * 60 / 2) + int(rate) * 60,
            ]
            data = preprocessing.scale(data, axis=1, with_std=True)

            for k in channelSet:
                hold = data[k, :]
                f, psd = spsig.welch(hold, fs=400, nperseg=2000)
                psd = np.nan_to_num(psd)
                psd /= psd.sum()
                for c in range(1, len(bands)):
                    featureList.append("BandEnergy_%i_%i_%i" % (j, k, c))
                    output.append(psd[(f > bands[c - 1]) & (f < bands[c])].sum())

        mydata.append(pd.DataFrame({"Features": output}, index=featureList).T)
        trainSample = pd.concat(mydata, ignore_index=True)

    new_outfile = outfile[:-4] + "_channel_{}_short.csv".format(channel_str)
    trainSample.to_csv(new_outfile)
    return 1


# Function to extract long-term features
def long_features(pat, outfile, datapath, channelSet, studyMode):
    f = datapath + "/*mat"

    pat_num = pat
    ff = glob.glob(f)

    label = [str(os.path.basename(n)) for n in ff]
    channel_str = ""
    for channel in channelSet:
        channel_str += "{}-".format(channel + 1)
    channel_str = channel_str.rstrip("-")
    output = []
    featureList = []

    bands = [0.1, 4, 8, 12, 30, 70]
    mydata = []

    for i in range(len(ff)):
        print("Running file number {}".format(i))
        output = []
        featureList = []
        if os.path.basename(ff[i]) == "1_45_1.mat":
            continue
        data = get_data(ff[i])
        data = preprocessing.scale(data, axis=1, with_std=True)
        featureList.append("File")
        output.append(label[i])
        featureList.append("pat")
        output.append(pat_num)

        if len(channelSet) == 16:
            # get correlation Coef. this will be 16x16
            h = np.corrcoef(data)
            h = np.nan_to_num(h)
            # only want upper triangle
            ind = np.triu_indices(16, 1)
            htri = h[ind]
            for ii in range(np.size(htri)):
                featureList.append("coef%i" % (ii))
                output.append(htri[ii])

            c, v = np.linalg.eig(h)
            c.sort()
            c = np.real(c)
            for e in range(len(c)):
                featureList.append("coef_timeEig%i" % (e))
                output.append(c[e])

        for j in channelSet:

            hold = spsig.decimate(data[j, :], 5, zero_phase=True)

            featureList.append("kurt%i" % (j))
            output.append(spstat.kurtosis(hold))

            featureList.append("skew%i" % (j))
            output.append(spstat.skew(hold))

            diff = np.diff(hold, n=1)
            diff2 = np.diff(hold, n=2)

            featureList.append("zerod%i" % (j))
            output.append(((diff[:-1] * diff[1:]) < 0).sum())

            featureList.append("RMS%i" % (j))
            output.append(np.sqrt((hold ** 2).mean()))

            f, psd = spsig.welch(hold, fs=80)
            psd[0] = 0

            featureList.append("MaxF%i" % (j))
            output.append(psd.argmax())

            featureList.append("SumEnergy%i" % (j))
            output.append(psd.sum())

            psd /= psd.sum()
            for c in range(1, len(bands)):
                featureList.append("BandEnergy%i%i" % (j, c))
                output.append(psd[(f > bands[c - 1]) & (f < bands[c])].sum())

            featureList.append("Mobility%i" % (j))
            output.append(np.std(diff) / hold.std())

            featureList.append("Complexity%i" % (j))
            output.append(np.std(diff2) * np.std(hold) / (np.std(diff) ** 2.0))

            # Extracting these following feature when running with Aretes' setting:
            if studyMode == 0:
                featureList.append("sigma%i" % (j))
                output.append(hold.std())

                featureList.append("zero%i" % (j))
                output.append(((hold[:-1] * hold[1:]) < 0).sum())

                featureList.append("sigmad1%i" % (j))
                output.append(diff.std())

                featureList.append("sigmad2%i" % (j))
                output.append(diff2.std())

                featureList.append("zerod2%i" % (j))
                output.append(((diff2[:-1] * diff2[1:]) < 0).sum())

                featureList.append("entropy%i" % (j))
                output.append(
                    -1.0 * np.sum(psd[f > bands[0]] * np.log10(psd[f > bands[0]]))
                )

        mydata.append(pd.DataFrame({"Features": output}, index=featureList).T)
        trainSample = pd.concat(mydata, ignore_index=True)

    new_outfile = outfile[:-4] + "_channel_{}_long.csv".format(channel_str)
    trainSample.to_csv(new_outfile)

    return 1


# Main function
def main():
    feat = json.load(open("SETTINGS.json"))
    pat = feat["pat"]
    study_mode = feat["study_mode"]
    if len(feat["channel_set"]) == 0:
        channel_set = [i for i in range(16)]
    else:
        channel_set = feat["channel_set"]

    outfile = "/Volumes/Samsung_T5/seizure_data/feature_data/{}_pat_{}_study_{}_modified.csv".format(
        feat["data_mode"], pat, study_mode
    )

    long_features(pat, outfile, feat["path"], channel_set, study_mode)

    # if study_mode == 0:
    #     short_features(pat, outfile, feat["path"], channel_set)


if __name__ == "__main__":
    main()
