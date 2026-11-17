"""Consistent journal figure rendering using open-source Matplotlib.

Matplotlib's STIX mathtext requires no system LaTeX installation. All figure
scripts export vector PDFs and high-resolution PNGs to ./outputs by default.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE=(.125,.405,.605)
COPPER=(.715,.315,.215)
PURPLE=(.445,.340,.610)
SLATE=(.30,.36,.41)
GREY=(.66,.67,.68)
LIGHT=(.78,.78,.78)


def configure():
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['Times New Roman','Times','STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix', 'font.size':12,
        'axes.labelsize':14, 'axes.linewidth':.8,
        'xtick.labelsize':11, 'ytick.labelsize':11,
        'xtick.direction':'in','ytick.direction':'in',
        'xtick.top':True,'ytick.right':True,
        'legend.frameon':False,'legend.fontsize':9.5,
        'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
        'path.simplify':True,
    })


def style(ax, ticks=11, grid=False):
    ax.tick_params(labelsize=ticks, direction='in', top=True,right=True,width=.8)
    for spine in ax.spines.values(): spine.set_visible(True)
    if grid: ax.grid(alpha=.11,which='both')
    else: ax.grid(False)


def letter(ax, name, xy=(.025,.97), size=14, color='black'):
    ax.text(xy[0],xy[1],f'({name})',transform=ax.transAxes,
            va='top',ha='left',fontsize=size,color=color)


def output_paths(name, root=None):
    dest=Path(root) if root else Path(__file__).resolve().parent/'outputs'
    dest.mkdir(parents=True,exist_ok=True)
    return dest/name


def export(fig,name,root=None,dpi=600):
    base=output_paths(name,root)
    fig.savefig(str(base)+'.pdf',format='pdf',bbox_inches='tight',pad_inches=.07)
    fig.savefig(str(base)+'.png',format='png',dpi=dpi,bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
    print(f'Exported {base}.pdf and {base}.png')
