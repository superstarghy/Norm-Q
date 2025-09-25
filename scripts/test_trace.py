import json
import matplotlib.pyplot as plt

# dict_keys(['schemaVersion', 'deviceProperties', 'record_shapes',
# 'profile_memory', 'traceEvents', 'traceName', 'displayTimeUnit', 'baseTimeNanoseconds'])

# class Node:
#     def __init__(self, cat, name, ts, dur):
#         self.cat = cat
#         self.name = name
#         self.ts = ts
#         self.dur = dur
#         self.children = list()
#         self.parent = None


# class TraceTree:
#     def __init__():
#         self.root = Node('root', 'root', 0, 0)
    
#     def insert(self, node, pnode=None):
#         if pnode == None:
#             pnode == self.root
#         if len(pnode.children) == 0:
#             pnode.append(node)
#         else:
#             for n in pnode.children:
#                 if(node.ts >= n.ts and node.ts + node.dur <= n.ts + n.dur):
#                     self.insert(node, n)

start = 526771880381.291 + 316500.921
mid = 526772277958.918
end = 526772277958.918 + 174333.561
print(f'Neuro-part: {(mid - start) / (end - start)}, Symbolic-part: {(end - mid) / (end - start)}')

'''
JSON_PATH = 'log/ctrlg/udc-an33-38_303867.1728890124251586592.pt.trace.json'

with open(JSON_PATH) as f:
    data = json.load(f)

ts0 = 526770803115.387 # us

# logits_processpr_call_2, 'ts': 526771880381.291, 'dur': 316500.921
# logits_processpr_call_3, 'ts': 526772277958.918, 'dur': 174333.561


event0 = {'cat': 'cpu_op', 'name': 'start', 'ts': 0, 'dur': 0}
trace = []
trace.append(event0)
for event in data['traceEvents']:
    if 'cat' in event.keys() and event['cat'] == 'cpu_op' and event['ts'] >= start and event['ts'] <= end:
        i = 0
        for t in range(len(trace)-1):
            if event['ts'] >= (trace[i]['ts'] + trace[i]['dur']) and (event['ts'] + event['dur']) <= trace[i + 1]['ts']:
                trace.insert(i + 1, event)
                i = len(trace)
                break
            elif event['ts'] <= trace[i+1]['ts'] and (event['ts'] + event['dur']) >= (trace[i+1]['ts'] + trace[i+1]['dur']):
                trace[i+1] = event
                i = len(trace)
                break
            elif event['ts'] >= trace[i+1]['ts'] and (event['ts'] + event['dur']) <= (trace[i+1]['ts'] + trace[i+1]['dur']):
                i = len(trace)
                break
            i += 1
        if i == len(trace) - 1:
            trace.append(event)

print(len(trace))
with open('NeSy.json', 'w') as file:
    json.dump(trace, file, indent=4)
'''

JSON_PATH = 'NeSy.json'

with open(JSON_PATH) as f:
    data = json.load(f)

# opname: cumulative_time
Ne_ops = {}
Sym_ops = {}
for event in data:
    if event['ts'] < mid:
        if event['name'] not in Ne_ops.keys():
            Ne_ops[event['name']] = event['dur']
        else:
            Ne_ops[event['name']] += event['dur']
    else:
        if event['name'] not in Sym_ops.keys():
            Sym_ops[event['name']] = event['dur']
        else:
            Sym_ops[event['name']] += event['dur']


print(Ne_ops.keys())
print(Sym_ops.keys())
Ne_ops = sorted(Ne_ops.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)[0:20]
Sym_ops = sorted(Sym_ops.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)[0:10]

labels1 = [t[0] for t in Ne_ops]
x1 = [t[1] for t in Ne_ops]
labels1.append('others')
x1.append(mid-start-sum(x1))

labels2 = [t[0] for t in Sym_ops]
x2 = [t[1] for t in Sym_ops]
labels2.append('others')
x2.append(end-mid-sum(x2))

explode=[0] * len(x1)
explode[0] = 0.25
plt.pie(
    x=x1,
    explode=explode,
    # labels=labels1,
    autopct='%.2f%%',
    pctdistance=1.25,
    textprops={'fontsize': 8}
)
plt.legend(labels1, loc='center right', fontsize='xx-small', title='Neuro', ncol=1, bbox_to_anchor=(-0.1, 0.5))
plt.savefig('fig/Ne_ops.', dpi=300)
plt.close()

explode=[0.6] * len(x2)
explode[0] = 0
plt.pie(
    x=x2,
    explode=explode,
    # labels=labels2,
    autopct='%.2f%%',
    pctdistance=1.25,
    textprops={'fontsize': 6},
)
plt.legend(labels2, loc="center right", fontsize='xx-small', title='Symbolic', bbox_to_anchor=(0.5, 0.5))
plt.savefig('fig/Sym_ops.png', dpi=300)
plt.close()
