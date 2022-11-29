
analyse目录下的csv文件是history_data.csv被处理后得到的，而该文件的数据来自iquant，时间为2020.1.1-2022.11.15。
命名格式为：pre_*_label_*_horizon_*.csv

correlation_analysis.py用于关联分析，只有old_feature和tushare文件夹下的数据做了关联分析。
关联分析结果被存放在两类文件中：
rst是pre表间做关联分析的结果
multi是pre表与nxt_close做分析的结果

文件夹说明：
v1_feature存放了v1特征的结果，horizon为5，label类型为2。
v2_tushare存放了v2特征的结果 horizon为5，label类型为2。
注意：
由于v1和v2使用特征的差异不影响预测目标nxt，故二者的nxt是一样的，因此只在v2_tushare中存了一次；
v1_feature中的数据也是来自tushare。

