"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

convenience functions to save plots
"""

import matplotlib.pyplot as plt

def save_plot(fig, path, spec='png'):
    '''
    Saves the plot to a file
    :param fig: figure to save
    :param path: path to the file
    :param spec: file format
    '''

    plt.tight_layout()
    if not path.startswith('results/'):
        path = 'results/' + path
    if not spec.startswith('.'):
        spec = '.' + spec
    if not spec in ['.png', '.pdf', '.svg']:
        raise Exception('unsupported file format')
    if spec == '.png':
        fig.savefig(path + spec, dpi=300)
    else:
        fig.savefig(path + spec)
