import matplotlib.pyplot as mp
import pandas as pd

def anaplot(sub,lab,org):
    fig, ax = mp.subplots(len(sub))
    for j in range(len(sub)):
        ax[j].plot(sub[j].to_numpy())
        ax[j].plot(org.to_numpy())
        ax[j].set_title(label=lab[j])
    fig.set_figheight(20)
    fig.set_figwidth(15)
    return fig

def anaplot2(sub,lab,org,lengh):
    fig, ax = mp.subplots(lengh,len(sub))
    for i in range(lengh):
        for j in range(len(sub)):
            ax[i][j].plot(sub[j].to_numpy()[:,i])
            ax[i][j].plot(org.to_numpy()[:,i])
            ax[i][j].set_title(label=lab[j] + sub[0].columns[i])
    fig.set_figheight(20)
    fig.set_figwidth(15)
    return fig

def getset(filename):
    gap = pd.read_csv(filename + "-gap.csv",skip_blank_lines=False,index_col=0)
    fill = pd.read_csv(filename + "-fill.csv",index_col=0)
    org = pd.read_csv(filename + "-data.csv",index_col=0)
    lengh = org.shape[1]
    print(lengh)
    if lengh == 1:
        fig = anaplot([fill,org],["fill","org"],gap)
    else:
        fig = anaplot2([fill,org],["fill","org"],gap,lengh)
    fig.savefig(filename + "-fig.pdf")

getset(input("set? "))
