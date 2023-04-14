
# Read the file results.txt into a numpy array, sort it by the second column,
# and graph it.
import numpy as np
import matplotlib.pyplot as plt

# Read the file results.txt into a numpy array. Treat it as a csv file.
data = np.genfromtxt('results.txt', delimiter=',')

# Sort the array by the second column
data = data[data[:,1].argsort()]

# Divide the second column by 1000 to get the time in milliseconds
data[:,1] /= 1000

# Graph the data as a connected line and enumerate the puzzles from 900 to 1000
plt.plot([x for x in range(900, 1000)], data[900:,1], 'k-')

# Annotate each point with the value of the first column
for i, txt in enumerate(data[900:,0]):
    plt.annotate(txt, (i+900, data[900+i,1]))

# Set the x and y axis labels
plt.xlabel('Puzzle #')
plt.ylabel('Time (us)')

# Show the graph
plt.show()
