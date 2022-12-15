import pandas as pd


def extract_nohup(dir_name, file_name, epoch, offset):
    if epoch > 0:
        end_epoch = epoch + offset
    count = 0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    train_p = []
    val_p = []
    path = f"../{dir_name}/{file_name}.log"
    with open(path, "r") as f:
        while True:
            l = f.readline()
            l.strip(' \n')
            if "train_loss" in l:
                idx = l.index("train_loss")
                train_loss = l[idx + 13:]
                train_loss_list.append(round(float(train_loss), 3))
                # print(f"train_loss:{train_loss}")
            if "dev_loss" in l:
                idx = l.index("dev_loss")
                dev_loss = l[idx + 10:]
                val_loss_list.append(round(float(dev_loss), 3))
                # print(f"val_loss:{dev_loss}")
            if "Dev Count" in l:
                count += 1
            # if "Binary Metric" in l:
            #     idx = l.index("Precession")
            #     end = l.index("Recall")
            #     p = l[idx + 12: end - 2] # 取小数点后4位
            #     if count % 2 == 0:
            #         val_p.append(round(float(p), 3))
            #     else:
            #         train_p.append(round(float(p), 3))
            if "Accuracy" in l:
                idx = l.index("Accuracy")
                acc = l[idx + 10:]
                if count % 2 == 0:
                    val_acc_list.append(round(float(acc), 3))
                    # print(f"val_acc:{acc}")
                else:
                    train_acc_list.append(round(float(acc), 3))
                    # print(f"train_acc:{acc}")
            if "Best Epoch" in l:
                break
            if epoch > 0 and "Epoch:" in l:
                idx = l.index("Epoch:")
                idx2 = l.index(",")
                now_epoch = l[idx + 7: idx2]
                if int(now_epoch) > end_epoch:
                    break
    data = [train_loss_list, val_loss_list, train_acc_list, val_acc_list
        # , train_p, val_p
            ]
    df = pd.DataFrame(data)
    df = df.T
    df.columns = ["train_loss", "val_loss", "train_acc", "val_acc"
        # , "train_p", "val_p"
                  ]
    # print(df.head())
    df.to_csv(f"./log/{file_name}.csv", index=False)


if __name__ == "__main__":
    pass
    # extract_nohup("../v2_feature_label_2_horizon_5.log")