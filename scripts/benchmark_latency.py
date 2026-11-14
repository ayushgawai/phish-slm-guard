#!/usr/bin/env python3
import sys, time, statistics
sys.path.insert(0,"src")
from phishguard.data import generate
from phishguard.model import FallbackScorer
if __name__=="__main__":
    s=FallbackScorer().fit(generate(40,seed=1)); msgs=generate(5,seed=3); times=[]
    for i in range(200):
        t=time.perf_counter(); s.predict_one(msgs[i%len(msgs)]); times.append((time.perf_counter()-t)*1000)
    times.sort(); print(f"n=200 p50={statistics.median(times):.2f}ms p99={times[int(.99*len(times))-1]:.2f}ms (CPU fallback/student scorer)")
