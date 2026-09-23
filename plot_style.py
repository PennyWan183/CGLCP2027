"""Canonical display names/colors for every predictive method; IDs remain unchanged."""
LABELS={
'pooled':'Pooled CP','local':'Local CP','global':'Global-rate plug-in','graph':'GraphLCP',
'pcs':'PCS Corrected','lpcs':'Localized PCS Corrected','cglcp':'CGLCP',
'rho':r'Oracle local ($\rho$)','alpha':r'Oracle local ($\alpha$)','safe':'Oracle-safe CP',
'weighted':'Weighted plug-in','envelope':'Weighted envelope'}
COLORS=dict(zip(LABELS,['#8296AA','#929292','#355C8C','#388C91','#A28BBC','#70518C','#E66B16','#278BC3','#67A9B4','#505866','#887F9D','#276B72']))
MARKERS=dict(zip(LABELS,['o','d','s','^','v','D','P','X','*','h','o','s']))
IDS={
'pooled_cp':'pooled','local_uncorrected':'local','encode_tau_uncorrected':'local',
'global_rate_plugin':'global','graph_lcp':'graph','pcs_sc_corrected':'pcs','pcs_corrected':'pcs','pcl900_global_reference_corrected':'pcs',
'localized_pcs_sc_corrected':'lpcs','localized_pcs_corrected':'lpcs','pcl900_tau_reference_corrected':'lpcs',
'cglcp':'cglcp','cglcp_certified':'cglcp','encode_tau_cglcp_sharp':'cglcp',
'oracle_local_rho':'rho','oracle_certificate_budget':'rho','oracle_rho':'rho',
'oracle_local_alpha':'alpha','oracle_alpha':'alpha','encode_tau_oracle_alpha':'alpha',
'oracle_safe':'safe','Weighted plug-in':'weighted','Weighted envelope':'envelope','CGLCP':'cglcp'}
ALIASES={label:key for key,label in LABELS.items()}
ALIASES.update({'Local, uncorrected':'local','Uncorrected local CP':'local','Graph-LCP':'graph','PCS-SC Corrected':'pcs','Localized PCS-SC Corrected':'lpcs',r'Localized PCS Corrected ($+\tau$)':'lpcs','Oracle local (alpha)':'alpha','Oracle local (rho budget)':'rho',r'Oracle local ($\rho$-budget)':'rho','Oracle certificate budget':'rho',r'Oracle local (same-$\rho$ budget)':'rho','Pooled':'pooled','Global':'global','PCS':'pcs',r'PCS+$\tau$':'lpcs',r'Oracle $\rho$':'rho',r'\textsc{CGLCP}':'cglcp'})
def key(x):return IDS.get(x,ALIASES.get(x))
def patch(module):
 for attr in ['LABEL','LABELS','METHOD_LABEL','METHOD_LABELS']:
  if hasattr(module,attr):
   d=getattr(module,attr)
   for k in list(d):
    if key(k):d[k]=LABELS[key(k)]
 for attr in ['COLOR','COLORS']:
  if hasattr(module,attr):
   d=getattr(module,attr)
   for k in list(d):
    if key(k):d[k]=COLORS[key(k)]
 for attr in ['MARK','MARKERS','MARKER']:
  if hasattr(module,attr):
   d=getattr(module,attr)
   for k in list(d):
    if key(k):d[k]=MARKERS[key(k)]
