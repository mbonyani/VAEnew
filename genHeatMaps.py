# This file generates heatmaps

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
#from sequenceModel import SequenceModel
from sequenceDataset import SequenceDataset

OALPHABET = ["A", "C", "G", "T"]

def genHeatmap(seq, out, name):
    fig, ax = plt.subplots(figsize=(10, 5))
    #colMapF = lambda x: x if abs(x) < 1 else math.copysign(math.log(abs(x)) + 1, x)
    #colMap = [[colMapF(manipOut[0, i, j]) for j in range(4)] for i in range(10)]
    im = ax.imshow(out[0, :, :], vmin=-2, vmax=2)

    #draw boxes around the correct pick
    correctCounter = 0
    for i in range(10):
        corectIndex = OALPHABET.index(seq[i])
        rect = plt.Rectangle((corectIndex-.5, i-.5), 1,1, fill=False, color="red", linewidth=4)
        ax.add_patch(rect)
        computedIndex = np.argmax(out[0, i, :])
        rect = plt.Rectangle((computedIndex-.5, i-.5), 1,1, fill=False, color="lime", linewidth=2)
        ax.add_patch(rect)
        if corectIndex == computedIndex:
            correctCounter += 1
    ax.set_title(str(correctCounter) + " / 10")

    #add color bar
    fig.colorbar(im, extend='both')
    #cbar.cmap = copy.copy(cbar.cmap)
    #cbar.cmap.set_over("red")
    #cbar.cmap.set_under("blue")
    ax.set_xticks(np.arange(len(OALPHABET)))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(OALPHABET)
    ax.set_yticklabels([seq[i] + "  " + str(i) for i in range(10)])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    #add labels to each box
    for i in range(10):
        for j in range(len(OALPHABET)):
            ax.text(j, i, "%.2f" % out[0, i, j], ha="center", va="center", fontsize="x-small", color="w")
    fig.savefig(name, bbox_inches='tight')

model = torch.load("./models/1623316111.pt")
dataset = SequenceDataset()
sequences = dataset.dataset['Sequence']
#the OG alex onehot encoding 1 liner
oheSeqs = dataset.transform_sequences(sequences.apply(lambda x: pd.Series([c for c in x])).to_numpy())
for n in range(0, 20):
    print("generated " + str(n))
    output, _, _, _, _ = model(torch.from_numpy(np.expand_dims(oheSeqs[n], 0)))
    genHeatmap(sequences[n], output.detach().numpy(), "./graphs/encodings/ohe" + str(n) + ".png")