import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import collections
import sys
guesses = collections.defaultdict(dict)

with open("datadump.txt") as f:
    lines = f.readlines()
    line = lines.pop(0)
    line = line.strip()
    while lines:
        if line.startswith("Source:"):
            line = line.split(' ')
            source = ' '.join(line[1:])
            source = source.strip()
            line = lines.pop(0)
            while not line.startswith("Source"):
                rest, percent = line.strip().split("Percentage: ")
                r = rest.replace("Guess: ", "")
                r = r.replace(",", "")
                r = r.strip()
                print(r)
                guesses[source][r] = float(percent)
                if lines:
                    line = lines.pop(0)
                else:
                    break
print(guesses)
SOURCES = ["Vox", "CNN", "Talking Points Memo", "Buzzfeed News", "Washington Post",
        "Guardian", "Atlantic", "Business Insider", "New York Times",
        "NPR", "Reuters", "New York Post", "Fox News", "National Review", "Breitbart"]
array = []
for s in SOURCES:
    d = []
    tot = sum(guesses[s].values())
    for other_s in SOURCES:
        if tot != 0 and other_s in guesses[s]:
            d.append(100 * round(guesses[s][other_s]/tot, 2))
        else:
            d.append(0)
    array.append(list(d))
df_cm = pd.DataFrame(array, SOURCES, SOURCES)
plt.figure(figsize = (20,20))
sn.set(font_scale=1.4)#for label size
x = sn.heatmap(df_cm, annot=True,annot_kws={"size": 20})# font size
fig = x.get_figure()
fig.savefig('ml_heatmap')
