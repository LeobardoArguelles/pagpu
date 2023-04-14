
# Read the file results.txt into a numpy array, sort it by the second column,
# and graph it.
import numpy as np
import matplotlib.pyplot as plt

# Read the file results.txt into a numpy array. Treat it as a csv file.
data = np.genfromtxt('results_16.txt', delimiter=',')

# Sort the array by the second column
data = data[data[:,1].argsort()]

# Divide the second column to convert from microseconds to seconds
data[:,1] = data[:,1] / 1000000

# Graph the data as a connected line and enumerate the puzzles from 900 to 1000
# plt.plot([x for x in range(900, 1000)], data[900:,1], 'k-')

# Graph the data as scatter points
plt.plot(["easy", "medium", "hard"], data[:,1], 'k.')

# Set the x and y axis labels
plt.xlabel('Puzzle #')
plt.ylabel('Time (s)')

# Show the graph
plt.show()
