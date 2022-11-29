1. data_generator_index.py 构建节点信息 如果修改了源文件，例如label，需要重新跑这个
2. graph_data_generator_index.py 构建子图结构 如果修改了图结构/划分比例，需要重新跑这个
3. data_process.py 构建节点/边特征属性 如果修改了正向反向边，需要重新跑这个
4. main.py 模型训练 注意修改图结构后，初始化时num要更改；以及label怎么分类，要从这里改dataset