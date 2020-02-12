import matplotlib.pylab as pylab
from cycler import cycler
import palettable
import seaborn as sns

params = {'text.usetex': True,
          'font.family': 'serif',
          'font.serif': ['CMU Serif'],
          'font.sans-serif': ['CMU Sans Serif'],
          'legend.fontsize': 17,
          'legend.fancybox': True,
          'legend.frameon': False,
          'legend.framealpha': 0.4,
          'legend.labelspacing': 0.5,
          'figure.figsize': (8 / 1.3, 6.5 / 1.3),
          'axes.labelsize': 19,
          'axes.titlesize':18,
          'axes.titlepad':7.5,
          'axes.linewidth':1.8,
          'axes.labelpad':10,
        #   'axes.prop_cycle': cycler('color', palettable.cartocolors.qualitative.Bold_10.hex_colors) + 
        #                     cycler(alpha=10*[.7]), 
          'axes.prop_cycle': cycler('color', sns.color_palette("Set1", 10).as_hex()) + 
                            cycler(alpha=10*[.7]), 
          'lines.linewidth':2.5,
          'xtick.labelsize':17.5,
          'ytick.labelsize':17.5,
          'xtick.top':True,
          'ytick.right':True,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'xtick.major.size': 7,
          'xtick.minor.size': 3,
          'ytick.major.size': 7,
          'ytick.minor.size': 3,
          'xtick.major.width': 1,
          'ytick.major.width': 1,
          'xtick.minor.width': 0.8,
          'ytick.minor.width': 0.8,
          'xtick.minor.visible': True,
          'ytick.minor.visible': True,
          'xtick.major.pad': 6,
          'ytick.major.pad': 6
         }
