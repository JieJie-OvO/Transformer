import os
import textgrid

def textgrid2outlist(srcpath):
    frames = {}
    files = [file for file in os.listdir(srcpath)]
    for path in files:
        frame = []
        name = path.split(".")[0]
        path = srcpath + "/" + path
        tgs = textgrid.TextGrid()
        # print(path)
        tgs.read(path)
        for i in range(len(tgs.tiers[0])):
            tg = tgs.tiers[0][i]
            if tg.mark == "" and i < len(tgs.tiers[0])-1:
                continue
            frame.append([tg.mark, tg.minTime ,tg.maxTime])
        frames[name] = frame
    return frames