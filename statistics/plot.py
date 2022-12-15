import argparse

import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
from extrct_nohup import extract_nohup


def plot(file):
    df = pd.read_csv(file)
    epoch = range(1, len(df) + 1)
    plt.figure(figsize=(20, 10), dpi=100)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for item in df.columns:
        idx = item.index("_")
        if "loss" in item:
            ax1.plot(epoch, df[item].to_numpy(), label=f'{item[:idx]}')
        elif "acc" in item:
            ax2.plot(epoch, df[item].to_numpy(), label=f'{item[:idx]}')
        # else:
        #     ax3.plot(epoch, df[item].to_numpy(), label=f'{item[:idx]}')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('acc')
    # ax3.set_ylabel("p")
    ax2.set_xlabel("epoch")
    idx = file.index("log/")
    title = f"{file[idx + 4:-4]}"
    ax1.set_title(title)
    ax1.legend()
    ax2.legend()
    # ax3.legend()
    plt.savefig(f"./pic/v2_feature/{title}.png")



if __name__ == "__main__":
    # for i in ['v1_feature_nohup']:
    #     paths = glob.glob(f"../{i}/*.log")
    #     for p in paths:
    #         path = re.sub(".log", '', p)
    #         path = re.sub(f"../{i}/", '', path)
    #         print(path)
    #         extract_nohup(i, path)
    #         plot(f"./log/{path}.csv")parser = argparse.ArgumentParser(description='Testing')
    #     parser.add_argument("--port", default=5000, type=int, help="port number")
    #     parser.add_argument('--device', type=str, default='cpu', help='If use gpu')
    #     parser.add_argument('--resume', type=str, default='weight.ckpt',
    #                         help='The path pf save model')  # 权重路径
    #     args = parser.parse_args()
    parser = argparse.ArgumentParser(description='Plot')
    parser.add_argument("--dir", default="", type=str, help="dir name")
    parser.add_argument('--file', type=str, help='file name')
    parser.add_argument('--epoch', type=int, default=0,
                        help='end epoch in x axis')
    parser.add_argument('--offset', type=int, default=5,
                        help='offset added on epoch')
    args = parser.parse_args()

    dir_name = args.dir
    path = args.file
    epoch = args.epoch
    offset = args.offset
    extract_nohup(dir_name, path, epoch, offset)
    plot(f"./log/{path}.csv")