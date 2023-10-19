"""
Author: Alex Hoffmann, 
Date: 10/19/2023
Description: File to plot the results of the magprime experiment
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def table():
    df = pd.read_csv("simulation_WAICUP_results.csv")
    snr_waic = np.mean(df["snr_waicup"])
    snr_mean = np.mean(df["snr_mean"])
    snr = np.array([np.fromstring(arg[1:-1], sep=',') for arg in df['snr_single'].to_numpy()])
    snr_min = np.mean(np.min(snr, axis=-1))

    rmse = np.array([np.fromstring(arg[1:-1], sep=',') for arg in df['rmse_single'].to_numpy()])
    min_rmse = np.mean(np.min(rmse, axis=-1))
    mean_rmse = np.mean(df["rmse_mean"])
    waicup_rmse = np.mean(df["rmse_waicup"])

    corr = np.array([np.fromstring(arg[1:-1], sep=',') for arg in df['corr_single'].to_numpy()])
    min_corr = np.mean(np.min(corr, axis=-1))
    mean_corr = np.mean(df["corr_mean"])
    waicup_corr = np.mean(df["corr_waicup"])

    print("Average &", np.round(mean_corr,4), "&", np.round(snr_mean,2), "&", np.round(mean_rmse,2), r"\\")
    print("Min &", np.round(min_corr,4), "&", np.round(snr_min,2), "&", np.round(min_rmse,2), r"\\")
    print("WAICUP &", np.round(waicup_corr,4), "&", np.round(snr_waic,2), "&", np.round(waicup_rmse,2), r"\\")

def pdfs():
    df = pd.read_csv("simulation_WAICUP_results.csv")
    snr_waic = df["snr_waicup"]
    snr_mean = df["snr_mean"]
    snr = np.array([np.fromstring(arg[1:-1], sep=',') for arg in df['snr_single'].to_numpy()])
    snr_min = np.min(snr, axis=-1)

    # Plot the distributions using distplot
    sns.distplot(snr_waic, hist=False, rug=False, label="WAICUP")
    sns.distplot(snr_mean, hist=False, rug=False, label="Average")
    sns.distplot(snr_min, hist=False, rug=False, label="Minimum")

    # Add labels and title
    plt.xlabel("SNR (dB)", fontsize = 14, fontdict={'weight': 'bold'})
    plt.ylabel("Probability density", fontsize = 14, fontdict={'weight': 'bold'})
    plt.title(" ", fontsize=16)
    plt.gca().tick_params(labelsize='large')
    plt.legend()

    # Show the plot
    plt.show()


def pdfs(axis = 0):
    df = pd.read_csv("magprime_results_A.csv")

    ica_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ica'].to_numpy()])[:,axis]
    mssa_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_mssa'].to_numpy()])[:,axis]
    ness_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ness'].to_numpy()])[:,axis]
    picog_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_picog'].to_numpy()])[:,axis]
    sheinker_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_sheinker'].to_numpy()])[:,axis]
    ream_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ream'].to_numpy()])[:,axis]
    ubss_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ubss'].to_numpy()])[:,axis]
    waicup_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_waicup'].to_numpy()])[:,axis]
    b1_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_b1'].to_numpy()])[:,axis]
    b2_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_b2'].to_numpy()])[:,axis]

    # Plot the distributions using distplot
    sns.distplot(ica_snr, hist=False, rug=False, label="ICA")
    sns.distplot(mssa_snr, hist=False, rug=False, label="MSSA")
    sns.distplot(ness_snr, hist=False, rug=False, label="NESS")
    sns.distplot(picog_snr, hist=False, rug=False, label="PiCoG")
    sns.distplot(sheinker_snr, hist=False, rug=False, label="Sheinker")
    sns.distplot(ream_snr, hist=False, rug=False, label="REAM")
    sns.distplot(ubss_snr, hist=False, rug=False, label="UBSS")
    sns.distplot(waicup_snr, hist=False, rug=False, label="WAICUP")
    sns.distplot(b1_snr, hist=False, rug=False, label="B1")
    sns.distplot(b2_snr, hist=False, rug=False, label="B2")

        # Add labels and title
    plt.xlabel("SNR (dB)", fontsize = 14, fontdict={'weight': 'bold'})
    plt.ylabel("Probability density", fontsize = 14, fontdict={'weight': 'bold'})
    plt.title(" ", fontsize=16)
    plt.gca().tick_params(labelsize='large')
    plt.legend()

    # Show the plot
    plt.show()

    
def rmseBoxPlot(axis = 0):
    df = pd.read_csv("magprime_results_A.csv")

    ica_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ica'].to_numpy()])[:,axis]
    mssa_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_mssa'].to_numpy()])[:,axis]
    ness_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ness'].to_numpy()])[:,axis]
    picog_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_picog'].to_numpy()])[:,axis]
    sheinker_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_sheinker'].to_numpy()])[:,axis]
    ream_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ream'].to_numpy()])[:,axis]
    ubss_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ubss'].to_numpy()])[:,axis]
    waicup_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_waicup'].to_numpy()])[:,axis]
    b1_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_b1'].to_numpy()])[:,axis]
    b2_rmse = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_b2'].to_numpy()])[:,axis]

    df = pd.DataFrame({"rmse_ica": ica_rmse, "rmse_mssa": mssa_rmse, "rmse_ness": ness_rmse, "rmse_picog": picog_rmse, "rmse_sheinker": sheinker_rmse, "rmse_ream": ream_rmse, "rmse_ubss": ubss_rmse, "rmse_waicup": waicup_rmse, "rmse_b1": b1_rmse, "rmse_b2": b2_rmse})

    fig, ax = plt.subplots() # use a smaller width value
    ax.boxplot(df[["rmse_ica", "rmse_mssa", "rmse_ness", "rmse_picog", "rmse_sheinker", "rmse_ream", "rmse_ubss", "rmse_waicup", "rmse_b1", "rmse_b2"]], flierprops=dict(marker='x', markersize=4)) # use flierprops to change the outliers to x's
    ax.set_ylabel("RMSE", fontsize=14, fontweight='bold') # set the y-axis label
    ax.set_xticklabels(["ICA", "MSSA", "NESS", "PiCoG", "Sheinker", "REAM", "UBSS", "WAICUP", "B1", "B2"], fontsize=16, fontweight='bold') # set the x-axis labels
    ax.set_title(" ", fontsize=16, fontweight='bold') # set the title
    ax.tick_params(labelsize='large')
    plt.show() # show the plot

"Create a box plot of the results of each algorithm"
def corrBoxPlot(axis = 0):
    df = pd.read_csv("magprime_results_A.csv")

    ica_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ica'].to_numpy()])[:,axis]
    mssa_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_mssa'].to_numpy()])[:,axis]
    ness_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ness'].to_numpy()])[:,axis]
    picog_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_picog'].to_numpy()])[:,axis]
    sheinker_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_sheinker'].to_numpy()])[:,axis]
    ream_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ream'].to_numpy()])[:,axis]
    ubss_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ubss'].to_numpy()])[:,axis]
    waicup_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_waicup'].to_numpy()])[:,axis]
    b1_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_b1'].to_numpy()])[:,axis]
    b2_corr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_b2'].to_numpy()])[:,axis]


    df = pd.DataFrame({"corr_ica": ica_corr, "corr_mssa": mssa_corr, "corr_ness": ness_corr, "corr_picog": picog_corr, "corr_sheinker": sheinker_corr, "corr_ream": ream_corr, "corr_ubss": ubss_corr, "corr_waicup": waicup_corr, "corr_b1": b1_corr, "corr_b2": b2_corr})

    fig, ax = plt.subplots() # use a smaller width value
    ax.boxplot(df[["corr_ica", "corr_mssa", "corr_ness", "corr_picog", "corr_sheinker", "corr_ream", "corr_ubss", "corr_waicup", "corr_b1", "corr_b2"]], flierprops=dict(marker='x', markersize=4)) # use flierprops to change the outliers to x's
    ax.set_ylabel("Correlation", fontsize=14, fontweight='bold') # set the y-axis label
    ax.set_xticklabels(["ICA", "MSSA", "NESS", "PiCoG", "Sheinker", "REAM", "UBSS", "WAICUP", "B1", "B2"], fontsize=16, fontweight='bold') # set the x-axis labels
    ax.set_title(" ", fontsize=16, fontweight='bold') # set the title
    ax.tick_params(labelsize='large')
    plt.show() # show the plot




def run():
    return

if __name__ == "__main__":
    run()