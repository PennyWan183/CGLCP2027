#!/usr/bin/env python3
"""Run the two score-level experiments and produce the publication plots."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import matplotlib.pyplot as plt
from simulation import score_robustness as suite
from simulation import local_necessity as local
from plot_figures import plot_local,plot_robustness
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--profile',choices=['smoke','paper'],default='smoke')
 p.add_argument('--only',choices=['both','local','robustness'],default='both')
 p.add_argument('--output-dir',type=Path)
 a=p.parse_args();out=a.output_dir or ROOT/'outputs'/a.profile;out.mkdir(parents=True,exist_ok=True)
 smoke=a.profile=='smoke';start=time.time()
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 if a.only in ['both','local']:
  d=out/'local_necessity';n=3 if smoke else 800
  print(f'Local necessity: {n} replicates per contamination setting',flush=True)
  oldargv=sys.argv
  try:
   # Preserve original numerical implementation; use the publication plotter below.
   sys.argv=['local_necessity.py','--repetitions',str(n),'--seed','1024','--alpha','0.1','--rho','0.02','--audit-size','30','--lcp-bandwidth','4.0','--output-dir',str(d)]
   local.main()
  finally:sys.argv=oldargv
  plot_local(d/'summary.csv',out/'figures')
 if a.only in ['both','robustness']:
  d=out/'score_robustness';d.mkdir(parents=True,exist_ok=True)
  n=3 if smoke else 4000;warmup=3 if smoke else 3000;rng=np.random.default_rng(20260831)
  print(f'Replaying S1 ({warmup} replicates per setting) to restore the original S3 RNG stream',flush=True)
  suite.run_s1(rng,warmup,alpha=.1,rho=.02)
  # Original S2 is deterministic, with no RNG consumption, so it can be skipped.
  print(f'Score robustness: {n} replicates per regime',flush=True)
  rows,summary=suite.run_s3(rng,n,alpha=.1,rho=.02)
  suite.write_csv(d/'replicates.csv',rows);suite.write_csv(d/'summary.csv',summary)
  (d/'config.json').write_text(json.dumps(dict(profile=a.profile,seed=1024,alpha=.1,rho=.02,s1_repetitions=warmup,s3_repetitions=n,rng_protocol='shared RNG: S1 then S3; deterministic S2 skipped'),indent=2)+'\n')
  plot_robustness(d/'summary.csv',out/'figures')
 (out/'run_metadata.json').write_text(json.dumps(dict(profile=a.profile,only=a.only,elapsed_seconds=time.time()-start,python=sys.version,numpy=np.__version__),indent=2)+'\n')
 print(f'Done: {out.resolve()}',flush=True)
if __name__=='__main__':main()
