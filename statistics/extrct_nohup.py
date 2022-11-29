import pandas as pd


def extract_nohup(file_name):
    count = 0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    with open(file_name, "r") as f:
        while True:
            l = f.readline()
            l.strip()
            if "train_loss" in l:
                idx = l.index("train_loss")
                train_loss = l[idx + 13:]
                train_loss_list.append(train_loss)
                # print(f"train_loss:{train_loss}")
            if "dev_loss" in l:
                idx = l.index("dev_loss")
                dev_loss = l[idx + 10:]
                val_loss_list.append(dev_loss)
                # print(f"val_loss:{dev_loss}")
            if "Accuracy" in l:
                count += 1
                idx = l.index("Accuracy")
                acc = l[idx + 10:]
                if count % 2 == 0:
                    val_acc_list.append(acc)
                    # print(f"val_acc:{acc}")
                else:
                    train_acc_list.append(acc)
                    # print(f"train_acc:{acc}")
            if "Best Epoch" in l:
                idx = l.index("Epoch-")
                epoch = l[idx + 6: idx + 7]
                break
    data = [train_loss_list, val_loss_list, train_acc_list, val_acc_list]
    df = pd.DataFrame(data)
    df = df.T
    df.columns = ["train_loss", "val_loss", "train_acc", "val_acc"]
    df.to_csv(f"./log/{file_name[:-4]}.csv", index=False)


if __name__ == "__main__":
    extract_nohup("../nohup_label_2_new_feature_old_relation.out")