#!/usr/bin/env python3
"""Convenience launcher for all open-source manuscript figures.

Default: Figures 2--5 (stationary calculations, fast).
Use --quick-fig1 for a non-publication Figure 1 smoke test.
Use --full-fig1 to additionally run the production full fP simulation.
"""
import argparse,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--quick-fig1',action='store_true')
    group.add_argument('--full-fig1',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    cmds=[]
    if args.full_fig1:cmds.append(['fig1.py'])
    elif args.quick_fig1:cmds.append(['fig1.py','--quick'])
    cmds.extend([[f'fig{i}.py'] for i in range(2,6)])
    for cmd in cmds:
        print('\n### Running:',' '.join(cmd),flush=True)
        subprocess.run([sys.executable,str(root/cmd[0]),*cmd[1:]],check=True)

if __name__=='__main__':main()
