import os
import glob
import pandas as pd
import shutil

folder = "/itet-stor/zrene/net_scratch/dataset_new/pile_no_contact"
ds = glob.glob(os.path.join(folder, "*/grasps.csv"))
out = os.path.join(folder, "../giga_pile_1M_no_contact")

all_dfs = []
for p in ds:
    print("OPENING", p)
    df = pd.read_csv(p)
    all_dfs.append(df)
    print("=>", out)

for f in glob.glob(os.path.join(folder, "*/*")):
    if os.path.isdir(f):
        print("DIR", f)
        for ff in glob.glob(os.path.join(f, "*")):
            target =  os.path.join(out, os.path.basename(os.path.dirname(ff)), os.path.basename(ff))
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
                
            if not os.path.exists(target):
                shutil.move(ff,  target)
            else:
                print("Can nopt move", ff, "to", target)
    else:
        if not os.path.exists(os.path.join(out, os.path.basename(f))):
            shutil.move(f, out)
        else:
            print("Not moving", f)  

#    print(f)
all_dfs = pd.concat(all_dfs)
all_dfs.to_csv(os.path.join(out, "grasps.csv"), index=False)
# print(all_dfs)