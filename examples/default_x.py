import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.labplotlib import plotting, Measurement
y = Measurement([1, 2, 4, 6, 8, 16, 32, 64, 128, 526, 1024, 2048])
plotting.line_plot(y)
plotting.plt.show()

#    if x is None:
#        x = range(1, len(y) + 1)