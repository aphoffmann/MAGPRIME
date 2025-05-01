"""
Author: Alex Hoffmann, 
Date: 10/19/2023
Description: File to plot the results of the magprime experiment
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import numpy as np
import magpylib as magpy
import scipy.spatial.transform as st


def polarPlot():
    df = pd.read_csv("magprime_results_A.csv")
    ica_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ica'].to_numpy()]), axis=0)
    mssa_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_mssa'].to_numpy()]), axis=0)
    ness_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ness'].to_numpy()]), axis=0)
    picog_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_picog'].to_numpy()]), axis=0)
    sheinker_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_sheinker'].to_numpy()]), axis=0)
    ream_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ream'].to_numpy()]), axis=0)
    ubss_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_ubss'].to_numpy()]), axis=0)
    waicup_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_waicup'].to_numpy()]), axis=0)
    b1_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_b1'].to_numpy()]), axis=0)
    b2_rmse = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['rmse_b2'].to_numpy()]), axis=0)

    ica_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ica'].to_numpy()]), axis=0)
    mssa_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_mssa'].to_numpy()]), axis=0)
    ness_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ness'].to_numpy()]), axis=0)
    picog_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_picog'].to_numpy()]), axis=0)
    sheinker_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_sheinker'].to_numpy()]), axis=0)
    ream_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ream'].to_numpy()]), axis=0)
    ubss_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_ubss'].to_numpy()]), axis=0)
    waicup_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_waicup'].to_numpy()]), axis=0)
    b1_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_b1'].to_numpy()]), axis=0)
    b2_corr = np.mean(np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['corr_b2'].to_numpy()]), axis=0)

    markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h"]
    labels = ["ICA", "MSSE", "NESS", "PiCoG", "SHEINKER", "REAM", "UBSS", "WAICUP", "BUS", "BOOM"]
    corr_arrays = [ica_corr, mssa_corr, ness_corr, picog_corr, sheinker_corr, ream_corr, ubss_corr, waicup_corr, b1_corr, b2_corr]
    rmse_arrays = [ica_rmse, mssa_rmse, ness_rmse, picog_rmse, sheinker_rmse, ream_rmse, ubss_rmse, waicup_rmse, b1_rmse, b2_rmse]

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, polar=True)

    # New mapping function
    def map_corr_to_radian(corr):
        return (1 - corr) * np.pi / 2

        # Looping over each correlation, rmse pair

    for i, (corr, rmse, label) in enumerate(zip(corr_arrays, rmse_arrays, labels)):
        # Applying the new function to the correlation data
        theta = map_corr_to_radian(corr)

        # Converting RMSE to log scale
        radius = np.log(rmse)/np.log(10)

        # Creating scatter plot for each axis with same color and label
        ax.scatter(theta, radius, label=label, marker=markers[i], s=50, alpha=0.8)

    # Setting theta axis limits for half polar plot
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(['1', '0', '-1'])
    ax.set_thetamin(90)
    ax.set_thetamax(0)

    plt.title("Half Polar plot of mean correlations and RMSEs")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()


def plotCubeSats():
    "Create Sources"
    d1 = magpy.current.Loop(current=10, diameter=2.0, orientation=st.Rotation.random(),  position=(-3.2, -2.0, 3.1))
    d2 = magpy.current.Loop(current=10, diameter=2.0, orientation=st.Rotation.random(), position=(-2, -3.7, 14.7))    
    d3 = magpy.current.Loop(current=10, diameter=2.0, orientation=st.Rotation.random(), position=(2.0, 3.6, 20.8))
    d4 = magpy.current.Loop(current=10, diameter=2.0, orientation=st.Rotation.random(), position=(1.4, 1.0, 23.7)) 
    src = [d1,d2,d3,d4]

    # Define CubeSat dimensions (assuming 10cm per U)
    unit = 10  # cm
    width, height, depth = unit, unit, 3 * unit  # 3U CubeSat dimensions

    # Function to plot a wireframe CubeSat centered at (0,0,0)
    def plot_centered_wireframe_cubesat(ax):
        # Define the half dimensions
        half_width, half_height, half_depth = width / 2, height / 2, depth / 2

        # Generate the cube data, 8 corners of a cube centered at (0,0,0)
        cube_definition = [
            (-half_width, -half_height, -0),
            (half_width, -half_height, -0),
            (half_width, half_height, -0),
            (-half_width, half_height, -0),
            (-half_width, -half_height, 2*half_depth),
            (half_width, -half_height, 2*half_depth),
            (half_width, half_height, 2*half_depth),
            (-half_width, half_height, 2*half_depth),
        ]
        cube_definition_array = [
            np.array(list(item))
            for item in cube_definition
        ]

        # Create a list of sides' polygons of our cube
        edges = [
            [cube_definition_array[0], cube_definition_array[1], cube_definition_array[2], cube_definition_array[3]],
            [cube_definition_array[4], cube_definition_array[5], cube_definition_array[6], cube_definition_array[7]], 
            [cube_definition_array[0], cube_definition_array[1], cube_definition_array[5], cube_definition_array[4]], 
            [cube_definition_array[2], cube_definition_array[3], cube_definition_array[7], cube_definition_array[6]], 
            [cube_definition_array[0], cube_definition_array[3], cube_definition_array[7], cube_definition_array[4]], 
            [cube_definition_array[1], cube_definition_array[2], cube_definition_array[6], cube_definition_array[5]],
        ]
        
        # Plot sides
        for edge in edges:
            ax.add_collection3d(Poly3DCollection([edge], facecolors='grey', linewidths=1, edgecolors='blue', alpha=0.1))

    # Set up dual panels
    fig = plt.figure(figsize=(10, 7))

    # Left Panel: CubeSat with gradiometry
    ax1 = fig.add_subplot(121, projection='3d')
    plot_centered_wireframe_cubesat(ax1)
    ax1.set_title('Gradiometry')

    # Right Panel: CubeSat with three bus-mounted magnetometers
    ax2 = fig.add_subplot(122, projection='3d')
    plot_centered_wireframe_cubesat(ax2)
    ax2.set_title('Bus-Mounted')

    # Add Magnetometers and noise
    s1 = magpy.Sensor(position=(5,5,30), style_size=1.7)
    s2 = magpy.Sensor(position=(-5,5,20), style_size=1.7)
    s3 = magpy.Sensor(position=(-5,-5,0), style_size=1.7)
    s = [s1,s2,s3]
    magpy.show(src, s, canvas=ax2)

    s1 = magpy.Sensor(position=(0,0,30), style_size=.5)
    s2 = magpy.Sensor(position=(0,0,60), style_size=.5)
    s = [s1,s2]
    magpy.show(src, s, canvas=ax1)


    # Set common parameters for both subplots and adjust view angles
    for ax in [ax1, ax2]:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 100])
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        ax.set_zlabel('Z [cm]')
        ax.set_box_aspect((1, 1, 3))
        ax.set_zticks(range(-10, 110, 20))
        ax.set_xticks(range(-10, 10, 5))
        ax.set_yticks(range(-10, 10, 5))
        ax.view_init(elev=30, azim=120)

    # Since multiple noise sources and magnetometers are plotted, 
    # we need to avoid repeating the labels in the legend. We do this by creating custom legend entries.
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Magnetometer'),
                    Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Noise Source')]

    def add_boom(ax):
        # Define the start and end points of the boom in cm
        boom_start = (0, 0, 30)
        boom_end = (0, 0, 60)
        
        # Plot the boom as a line
        ax.plot([boom_start[0], boom_end[0]],
                [boom_start[1], boom_end[1]],
                [boom_start[2], boom_end[2]], 'k-', linewidth=2)
    add_boom(ax1) 




    plt.tight_layout()
    plt.show()

def pdfSubplotsA():
    # Load the data
    df = pd.read_csv("magprime_results_A.csv")

    # Parse data arrays
    ica_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ica'].to_numpy()])
    mssa_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_mssa'].to_numpy()])
    ness_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ness'].to_numpy()])
    picog_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_picog'].to_numpy()])
    sheinker_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_sheinker'].to_numpy()])
    ream_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ream'].to_numpy()])
    ubss_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ubss'].to_numpy()])
    waicup_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_waicup'].to_numpy()])
    b1_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_b1'].to_numpy()])
    b2_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_b2'].to_numpy()])

    # Create the plot
    fig, axes = plt.subplots(5, 2, figsize=(15, 10), sharex=True, sharey=True, dpi = 200)
    fig.suptitle("Distribution of SNR in Gradiometry Configuration", fontsize=22, y=.98)

    # List of algorithm labels
    algorithms = ["ICA", "MSSA", "NESS", "PiCoG", "Sheinker", "REAM", "WAICUP", "UBSS", "BUS", "BOOM"]

    # List of data arrays
    data_arrays = [ica_snr, mssa_snr, ness_snr, picog_snr, sheinker_snr, ream_snr, waicup_snr, ubss_snr, b1_snr, b2_snr]

    # Colors for each subplot
    linestyles = ['dashed', 'solid', 'dotted']
    colors = sns.color_palette("husl", len(algorithms))
    axs = ["X", "Y", "Z"]

    # Plot each algorithm's SNR distribution in a separate subplot
    for i, (data, algorithm) in enumerate(zip(data_arrays, algorithms)):
        row, col = divmod(i, 2)
        for axis in range(3):
            sns.kdeplot(data[:, axis], color=colors[i], linestyle=linestyles[axis], ax=axes[row, col], label=f"{axs[axis]}-Axis",)
        axes[row, col].set_xlabel("SNR (dB)", fontsize=16)
        axes[row, col].set_ylabel("Density", fontsize=16)
        axes[row, col].text(0.5, 0.95, f"{algorithms[i]}", transform=axes[row, col].transAxes, fontsize=14, fontweight='bold', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        axes[row, col].grid()
        axes[row, col].tick_params(axis='both', which='major', labelsize=12)
        axes[row, col].set_xlim(-61,61)
        #axes[row, col].legend( fontsize=14, loc =1)

    lines = [plt.Line2D([0], [0], color='black', linestyle=linestyle) for linestyle in linestyles]
    labels = [f"{axis}-Axis" for axis in axs]
    fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(top=0.9)

    # Show the plot
    plt.show()

def pdfSubplotsB():
    # Load the data
    df = pd.read_csv("magprime_results_B.csv")

    # Parse data arrays
    ica_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ica'].to_numpy()])
    mssa_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_mssa'].to_numpy()])
    ness_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ness'].to_numpy()])
    picog_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_picog'].to_numpy()])
    sheinker_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_sheinker'].to_numpy()])
    ream_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ream'].to_numpy()])
    ubss_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_ubss'].to_numpy()])
    waicup_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_waicup'].to_numpy()])
    b1_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_b1'].to_numpy()])
    b2_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_b2'].to_numpy()])
    b3_snr = np.array([np.fromstring(arg[1:-1], sep=' ') for arg in df['snr_b3'].to_numpy()])

    # Create the plot
    fig, axes = plt.subplots(6, 2, figsize=(15, 10), sharex=True, sharey=True, dpi = 200)
    fig.suptitle("Distribution of SNR in Boomless Configuration", fontsize=22, y=.98)

    # List of algorithm labels
    algorithms = ["ICA", "MSSA", "NESS", "PiCoG", "Sheinker", "REAM", "WAICUP", "UBSS", "M1", "M2", "M3"]

    # List of data arrays
    data_arrays = [ica_snr, mssa_snr, ness_snr, picog_snr, sheinker_snr, ream_snr, waicup_snr, ubss_snr, b1_snr, b2_snr, b3_snr]

    # Colors for each subplot
    linestyles = ['dashed', 'solid', 'dotted']
    colors = sns.color_palette("husl", len(algorithms))
    axs = ["X", "Y", "Z"]

    # Plot each algorithm's SNR distribution in a separate subplot
    for i, (data, algorithm) in enumerate(zip(data_arrays, algorithms)):
        row, col = divmod(i, 2)
        for axis in range(3):
            sns.kdeplot(data[:, axis], color=colors[i], linestyle=linestyles[axis], ax=axes[row, col], label=f"{axs[axis]}-Axis",)
        axes[row, col].set_xlabel("SNR (dB)", fontsize=16)
        axes[row, col].set_ylabel("Density", fontsize=16)
        axes[row, col].text(0.5, 0.95, f"{algorithms[i]}", transform=axes[row, col].transAxes, fontsize=14, fontweight='bold', ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        axes[row, col].grid()
        axes[row, col].tick_params(axis='both', which='major', labelsize=12)
        axes[row, col].set_xlim(-61,61)
        #axes[row, col].legend( fontsize=14, loc =1)

    axes[-1,-1].set_visible(False); axes[4,1].tick_params(axis='x', which='both', labelbottom=True)
    axes[4, 1].set_xlabel("SNR (dB)", fontsize=14)
    lines = [plt.Line2D([0], [0], color='black', linestyle=linestyle) for linestyle in linestyles]
    labels = [f"{axis}-Axis" for axis in axs]
    fig.legend(lines, labels, loc='lower center', fontsize=16, ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(top=0.9)

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

    fig, ax = plt.subplots(figsize=(10,5)) # use a smaller width value
    ax.boxplot(df[["rmse_ica", "rmse_mssa", "rmse_ness", "rmse_picog", "rmse_sheinker", "rmse_ream", "rmse_ubss", "rmse_waicup", "rmse_b1", "rmse_b2"]], flierprops=dict(marker='x', markersize=4)) # use flierprops to change the outliers to x's
    ax.set_ylabel("RMSE", fontsize=14, fontweight='bold') # set the y-axis label
    ax.set_xticklabels(["ICA", "MSSA", "NESS", "PiCoG", "Sheinker", "REAM", "UBSS", "WAICUP", "B1", "B2"], fontsize=16, fontweight='bold') # set the x-axis labels
    ax.set_title(" ", fontsize=16, fontweight='bold') # set the title
    ax.tick_params(labelsize='large')
    ax.set_yscale('log')
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

    fig, ax = plt.subplots(figsize=(10,5)) # use a smaller width value
    ax.boxplot(df[["corr_ica", "corr_mssa", "corr_ness", "corr_picog", "corr_sheinker", "corr_ream", "corr_ubss", "corr_waicup", "corr_b1", "corr_b2"]], flierprops=dict(marker='x', markersize=4)) # use flierprops to change the outliers to x's
    ax.set_ylabel("Correlation", fontsize=14, fontweight='bold') # set the y-axis label
    ax.set_xticklabels(["ICA", "MSSA", "NESS", "PiCoG", "Sheinker", "REAM", "UBSS", "WAICUP", "B1", "B2"], fontsize=16, fontweight='bold') # set the x-axis labels
    ax.set_title(" ", fontsize=16, fontweight='bold') # set the title
    ax.tick_params(labelsize='large')
    plt.show() # show the plot

def createTable():
    # Load the data
    df = pd.read_csv('magprime_results_C.csv')

    # Function to convert string representations of arrays into numpy arrays
    def str_to_nparray(data_series):
        return np.array([np.fromstring(item[1:-1], sep=' ') for item in data_series])

    # Creating a dictionary to hold the median results for each metric and axis
    median_results = {}

    # List of algorithms based on the provided keys, now including 'b3'
    algorithms = [
        'ica', 'mssa', 'ness', 'picog',
        'sheinker', 'ream', 'ubss', 'waicup', 'b1', 'b2', 'b3'
    ]

    # List of metrics
    metrics = ['rmse', 'corr', 'snr']

    # Iterate over each metric and algorithm to compute the median for each axis
    for metric in metrics:
        for axis in ['x', 'y', 'z']:
            for algorithm in algorithms:
                key = f"{metric}_{algorithm}"
                # Convert the string values to numpy arrays
                data_np = str_to_nparray(df[key])
                # Compute the median for the current axis (0, 1, 2 for x, y, z respectively)
                axis_index = ['x', 'y', 'z'].index(axis)
                median_results[f"{metric}_{axis}_{algorithm}"] = np.median(data_np[:, axis_index], axis=0)

    # Initialize an empty DataFrame with the multi-index for the final table
    multi_index = pd.MultiIndex.from_tuples(
        [(metric, f"{axis}_axis") for metric in metrics for axis in ['x', 'y', 'z']],
        names=['Metric', 'Axis']
    )
    median_table = pd.DataFrame(index=multi_index, columns=algorithms)

    # Populate the DataFrame with median values
    for metric in metrics:
        for axis in ['x', 'y', 'z']:
            for algorithm in algorithms:
                median_table.loc[(metric, f"{axis}_axis"), algorithm] = median_results[f"{metric}_{axis}_{algorithm}"]

    print(median_table)

