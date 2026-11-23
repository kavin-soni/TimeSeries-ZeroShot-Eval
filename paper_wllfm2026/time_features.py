"""Router overhead: feature-extraction latency (Section IV-C and the cost model).

Usage: python paper_wllfm2026/time_features.py [path/to/extract_series_features.py]

Times the repo's own feature functions on synthetic series of Traffic length
(14,035 points), 1,000 points and the shortest M4 Daily history (93 points).
The paper reports about 0.9 ms (M4 length) and 2.7 ms (Traffic length) on one
CPU core of the build machine; expect machine-to-machine variation.
"""
import os
import importlib.util, time, numpy as np, sys
path=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results','extract_series_features.py')
spec=importlib.util.spec_from_file_location('ex',path); ex=importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
rng=np.random.default_rng(0)
feats=lambda x,S:(ex.spectral_entropy(x),ex.cv(x),ex.seasonal_autocorr(x,S),ex.trend_strength(x))
for n,S in ((14035,168),(1000,7),(93,7)):
    x=(np.sin(np.arange(n)*2*np.pi/S)+rng.normal(0,.3,n)+5).astype(np.float32)
    for _ in range(20): feats(x,S)
    t=[]
    for _ in range(300):
        a=time.perf_counter(); feats(x,S); t.append(time.perf_counter()-a)
    t=np.array(t)*1e3; print(n,S,'median ms %.3f  p95 ms %.3f'%(np.median(t),np.percentile(t,95)))
