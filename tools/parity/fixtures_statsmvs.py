import json, math, numpy as np, scipy, scipy.stats as st
def enc(v):
    v=float(v)
    if math.isinf(v): return "inf" if v>0 else "-inf"
    if math.isnan(v): return "nan"
    return v
cases=[]
def add(data,alpha):
    data=[float(v) for v in data]
    m,v,s=st.bayes_mvs(data,alpha=alpha)
    cases.append({"fn":"bayes_mvs","data":data,"alpha":alpha,
        "expect":{"mean":[enc(m.statistic),[enc(m.minmax[0]),enc(m.minmax[1])]],
                  "var":[enc(v.statistic),[enc(v.minmax[0]),enc(v.minmax[1])]],
                  "std":[enc(s.statistic),[enc(s.minmax[0]),enc(s.minmax[1])]]}})
add([6,9,12,7,8,8,13],0.90)
add([1.0,2.0,3.0,4.0,5.0,6.5,7.2,1.1],0.95)
add([2.5,2.5,2.5,2.5,3.5],0.99)
add(list(np.linspace(-3,3,20)),0.90)
add([0.1,0.2],0.80)
add([10,20,30,40,50,60,70,80,90],0.95)
add(list(np.arange(1,13)),0.5)
add([-1.5,0.0,1.5,3.0,-2.2,4.4,0.9,1.1,2.2,3.3],0.90)
out={"module":"stats","scipy":scipy.__version__,"statsmvs_cases":cases}
json.dump(out,open("../../tests/fixtures/parity/stats_mvs.json","w"),indent=1)
print("wrote",len(cases))
