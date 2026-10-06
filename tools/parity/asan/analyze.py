import json,collections,sys,re
groups=collections.OrderedDict()
other=[]
for f in sys.argv[1:]:
  for l in open(f):
    r=json.loads(l)
    t=r['type']
    if t=='PART': continue
    if t=='SAN':
        h=r['head']
        m=re.match(r'(\S+:\d+):\d+: runtime error: (.*)',h)
        if m:
            key=('UB',m.group(1), re.sub(r'[-\d.e+]+','N',m.group(2))[:60])
        else:
            fr=[x for x in r['frames'] if 'csrc' in x or 'cxx/' in x or 'src/' in x or 'third_party' in x]
            mm=re.search(r' in (.+) (\S+)$', fr[0]) if fr else None; loc=mm.groups() if mm else ('?',fr[0] if fr else '?')
            kind=re.search(r'AddressSanitizer: (\S+)',h)
            key=('ASAN',kind.group(1) if kind else h[:40],loc[1])
        g=groups.setdefault(key,[])
        g.append(r)
    else:
        other.append(r)
for k,g in groups.items():
    fs=sorted(set(x['method'] for x in g))
    r=g[0]
    print('==',k, 'n=%d'%len(g), 'fns=',','.join(fs[:12]))
    print('   ex:',r['method'],r['case'],r['desc'][:120])
    print('   ', r['head'][:200])
    for fr in r['frames'][:4]: print('     ',fr[:160])
print()
oc=collections.defaultdict(list)
for r in other: oc[(r['type'],r['method'])].append(r)
for k,g in oc.items():
    print(k, len(g), '; '.join(f"{r['case']}:{r['desc'][:50]}{(' rc='+str(r.get('rc'))) if 'rc' in r else ''}{(' %.0fs'%r['secs']) if 'secs' in r else ''}" for r in g[:4]))
