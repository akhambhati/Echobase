import numpy as np
import matplotlib.pyplot as plt
import sys


def update_rcparams(rcParams):
    params = {
              ## Fonts
             'font.family': 'sans-serif',
             'font.style': 'normal',
             'font.variant': 'normal',
             'font.weight': 'medium',
             'font.stretch': 'normal',
             'font.size': 12.0,
             'font.serif': ['Times', 'Palatino', 'New Century Schoolbook', 'Bookman', 'Computer Modern Roman'],
             'font.sans-serif': ['Arial', 'Helvetica', 'Avant Garde', 'Computer Modern Sans serif'],
             'font.cursive': ['Zapf Chancery', 'Sand', 'cursive'],
             'font.fantasy': ['Comic Sans MS', 'Chicago', 'Charcoal', 'Impact', 'Western', 'fantasy'],
             'font.monospace': ['Source Code Pro', 'Courier', 'Computer Modern Typewriter'],
        
              ## Lines
              'lines.linewidth': 2.0,
              'lines.linestyle': '-',
              'lines.markeredgewidth': 0.0,
              'lines.markersize': 6.0,
              'lines.solid_joinstyle' : 'round',
              'lines.solid_capstyle': 'round',
              'lines.antialiased': True,        
                 
              ## Patches
              'patch.linewidth': 1.0,
              'patch.facecolor': [0.75, 0.75, 0.75],
              'patch.edgecolor': [0.5, 0.5, 0.5],             
              'patch.antialiased': True,
        
              ## Axes
              'axes.facecolor': 'white',
              'axes.grid': True,
              'axes.titlesize': 24.0,
              'axes.labelsize': 18.0,
              'axes.axisbelow': True,
              'axes.formatter.limits': [-7, 7],
              'axes.formatter.use_mathtext': True,
              'axes.xmargin': 0,
              'axes.ymargin': 0,
              'axes.spines.top': False,
              'axes.spines.right': False,
              'axes.color_cycle': ['5DA5DA', 'FAA43A', '60BD68', 'F17CB0', 'B2912F', 'B276B2', 'DECF3F', 'F15854', '4D4D4D'],
              'polaraxes.grid': True,
           
              ## Ticks
              'xtick.major.size': 4,
              'xtick.minor.size': 2,
              'xtick.major.width': 1,
              'xtick.minor.width': 1,
              'xtick.major.pad': 6,
              'xtick.direction': 'out',        
        
              'ytick.major.size': 4,
              'ytick.minor.size': 2,
              'ytick.major.width': 1,
              'ytick.minor.width': 1,
              'ytick.major.pad': 6,
              'ytick.direction': 'out',
        
              ## Grids
              'grid.color': [0.7, 0.7, 0.7],
              'grid.linestyle': 'solid',
              'grid.alpha': 0.2,
        
              ## Figure
              'figure.figsize': [8, 6],
              'figure.dpi': 150,
              'figure.facecolor': 'white',
              'figure.autolayout': True,
              'figure.max_open_warning': 40,

              ## Savingz
              'savefig.dpi': 300,
              'savefig.format': 'svg',
              'savefig.bbox': 'tight',
              'savefig.pad_inches': 0.1,
              'savefig.jpeg_quality': 95,
              
              ## Verbose
              'verbose.level': 'helpful',
              'verbose.fileo': sys.stdout
    }

                
    rcParams.update(params)

    return rcParams


def set_box_color(bp, color, fcolors):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
    for patch, fliers, fcolor in zip(bp['boxes'], bp['fliers'], fcolors):
        patch.set_facecolor(fcolor)
        plt.setp(fliers, markerfacecolor=fcolor, color=fcolor, marker='o', markersize=3)        

    return bp

