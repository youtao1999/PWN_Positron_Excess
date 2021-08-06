'''
For each channel, this python script loops through the chisq vs cross section data for each mass,
calculates the delta chisq for this mass, and from this delta chisq calculates the statistical significance
for this mass value. Plot the statistical significance vs mass for each channel.

work flow:
    1. Loop through each chisq vs cross section data file
    2. Calculate delta chisq for each mass
    3. Calculate statistical significance for each mass
    4. Store the data of mass vs statistical significance for this channel
    5. Output the plot for mass vs statistical significance for this channel, do the same for all channels.
'''