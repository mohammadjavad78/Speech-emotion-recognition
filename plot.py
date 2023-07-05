dict={}

loss=(-2,'Loss')
acc=(-1,'Accuracy')

calc=acc
names='hubert2'
with open(f"{names}.csv",'r') as f:
    lines=f.readlines()
for i in range(5):
    dict[i+1]={}
for line in lines[1:]:
    name=line.split(',')[2].split(',')[0]
    i=line.split('original,')[1].split(',')[0]
    if(name=="train"):
        dict[int(i)][name]=float(line.split(',')[calc[0]-4])
    if(name=="test"):
        dict[int(i)][name]=float(line.split(',')[calc[0]].split('\n')[0])
    if(name=="val"):
        dict[int(i)][name]=float(line.split(',')[calc[0]-2])

print(names,dict)
import matplotlib.pyplot as plt
trains=[]
for i in range(5):
    trains.append(dict[i+1]["train"])
plt.plot([i+1 for i in range(5)],trains,label="train")
# plt.show()


trains=[dict[i+1]['test'] for i in range(5)]
plt.plot([i+1 for i in range(5)],trains,label="test")
# plt.show()


trains=[dict[i+1]['val'] for i in range(5)]
plt.plot([i+1 for i in range(5)],trains,label="val")

plt.title(f"{calc[1]}_{names}")
plt.xlabel("Epochs")
plt.ylabel(f"{calc[1]}")
plt.legend()
plt.show()