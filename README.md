# polarization_grainsize
#Impact of grain size on linear polarization in particulate materials
#These python scripts are used to process the hyperspectral polarization data on various particulate materials used in this repository to analyze the impact of grain size.
# The file paths ar often hard coded and may need to be adjsuted based on where the data is stored for post-processing scripts. The main script allows you to choose the
# desired directory of data to process.

#Order of scripts:
# 1. Generalized_Polarization_Modeling.py is the main script, used to process the raw imagery into CSV files of line averaged data for every wavelength and phase angle. 
# Use this script first. This must be done separately for each sample.

# 2. Combine_Samples.py processes different grain size categories for each different soil sample to compare the polarization curves of different grain sizes. They also
# do 1st, 3rd, and 5th order polynomial fits to the data. This must be done separately for each sample.

# 3. RMSE_analysis.py and RMSE_nephsmall.py look at the root mean squared error of the polynomial fits to the data in its full range and within the 20-80 deg phase angle
# linear region. The Nephsmall one is for the 1-5 micron nepheline specifically since the files work a little differently. This must be done separately for each sample.

# Slope_Table.py and Slope_analysis.py look at the slope of the linear region of the polarization curve when fit to a line for all samples.
# This must be done separately for each sample.

#Gransize_Slope_2022.py builds the plots showing grain size vs the slope of the linear region of the polarization curve for each sample.
#This must be done separately for each sample.
