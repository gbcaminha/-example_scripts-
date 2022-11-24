#!/usr/bin/env python3
# ==================================
# Author: Gabriel Bartosch Caminha - gbcaminha@gmail.com
# ==================================

"""
Simple example to plot 2d distributions from a parameter chain
"""

import argparse

from matplotlib import rc
from scipy.stats import multivariate_normal
from getdist import MCSamples


import matplotlib.pyplot as plt
import numpy as np

rc('text', usetex=True)
font = {'family' : 'serif', \
        'size' : 18}
rc('font', **font)

def main_plot_2d_distribution():
    """
    run python plot_2d_distribution.py -h for description
    Main function: creates a 2-dimensional distribution and computes the PDFs
    using numpy.histogram2d and getdist.MCSamples. The numpy.histogram2d
    generates a noisy 2d PDF and getdist.MCSamples a smooth PDF.
    Input
     - no inputs
    Output
     - None
    """

	# argument parser for command line
    parser = argparse.ArgumentParser(\
        description = "Program to create smooth contour figures from 2d" + \
                      " chains.", \
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("npts", help = "Number of points", type=int)
    parser.add_argument("-bins", help = "Number of bins", type=int, default=100)
    args = parser.parse_args()


	# covariance matrix for 2d gaussian 1
    covariance1 = [[1.0, 0.5], \
                   [0.5, 0.5]]
    guassian2d_1 = multivariate_normal(mean=[0,0], cov=covariance1)
    # pseudo sampling of gaussian 1 MCMC like chain
    dist1 = guassian2d_1.rvs(size=args.npts).transpose()

	# covariance matrix for 2d gaussian 2
    covariance2 = [[2.0, -0.6], \
                   [-0.6, 0.3]]
    guassian2d_2 = multivariate_normal(mean=[1,2], cov=covariance2)
    # pseudo sampling of gaussian 2 MCMC like chain
    dist2 = guassian2d_2.rvs(size=args.npts).transpose()

	# concatenate both 'chains' to obtain a more complex distribution
    dist_combined_x = np.concatenate([dist1[0], dist2[0]])
    dist_combined_y = np.concatenate([dist1[1], dist2[1]])

	# creates a numpy 2D histogram
    hist2d_numpy = np.histogram2d(dist_combined_x, dist_combined_y, \
                                  bins=args.bins, density=True)

	# obtains the limits of the histogram for future calculations
    limits = [hist2d_numpy[1].min(), hist2d_numpy[1].max(), \
              hist2d_numpy[2].min(), hist2d_numpy[2].max()]

	# define levels for plt.contourf figure. See end of the script
    levels_np = np.linspace(hist2d_numpy[0].min(), hist2d_numpy[0].max(), 10)

    #--------------------------------------------------------------------------
    #--- Now using getdist package
    #--------------------------------------------------------------------------
	# define ranges in a dictionary for MCSamples input
    ranges = {"x":limits[0:2], "y":limits[2:4]}
    sample_gd = MCSamples(samples=[dist_combined_x, dist_combined_y], \
                          names=["x", "y"], ranges=ranges)
    # Load method to compute the PDF 
    density2d = sample_gd.get2DDensityGridData(j="x", j2="y", get_density=True)

	# Computes bin width 
    delta_x = hist2d_numpy[1][1] - hist2d_numpy[1][0]
    delta_y = hist2d_numpy[2][1] - hist2d_numpy[2][0]

	# Compute PDF at the central position of each bin in a regular grid
	# density_grid will be a matrix with dimension args.bins X args.bins
    density_grid = density2d(hist2d_numpy[1][0:-1] + delta_x/2, \
                             hist2d_numpy[2][0:-1] + delta_y/2)

	# define levels for plt.contourf figure, now using PDF computed with getdist
    levels_getdist = np.linspace(density_grid.min(), density_grid.max(), 10)

	# From here the code will make the figure to compare the original points,
	# the PDF computed with numpy.histogram2d and PDF computed with getdist
    plt.figure(figsize=(12, 3.5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(dist_combined_x, dist_combined_y, marker=".", markersize=1, \
             linestyle= 'None', alpha=0.1)
    ax1.set_title("original chain")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlim(ranges["x"])
    ax1.set_ylim(ranges["y"])


    ax2 = plt.subplot(1, 3, 2)
    ax2.contourf(hist2d_numpy[0].transpose(), extent=limits, levels=levels_np)
    ax2.set_title("np hist2d density")
    ax2.set_xlabel("X")
    ax2.set_xlim(ranges["x"])
    ax2.set_ylim(ranges["y"])


    ax3 = plt.subplot(1, 3, 3)
    ax3.contourf(density_grid.transpose(), extent=limits, levels=levels_getdist)
    ax3.set_title("getdist density")
    ax3.set_xlabel("X")
    ax3.set_xlim(ranges["x"])
    ax3.set_ylim(ranges["y"])

    plt.savefig("comparison.png", bbox_inches="tight", dpi=200)

if __name__ == '__main__':


    main_plot_2d_distribution()
