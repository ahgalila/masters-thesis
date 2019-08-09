import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = 5

data = (100, 60.16, 82.63, 84.20, 85.20)
errors = (0, 4.5, 3.7, 2.1, 2.7)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

rects1 = ax.bar(index, data, bar_width, yerr=errors)

ax.set_xlabel('Method')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Experiment Results')
ax.set_xticks(index)
ax.set_xticklabels(('Method A', 'Method B', 'Method C', 'Method D', 'Method E'))

fig.tight_layout()
plt.show()