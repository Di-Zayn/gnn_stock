import pandas as pd
from flask import Flask, request, jsonify
import torch
import sys

sys.path.append("../")
from config import Config
from data_process.data_script import RELATION_NUM
from api.get_graph import HGTModule
from api.get_graph import get_graph

app = Flask(__name__)
config = Config()

def init_model():
    model_save_path = f"../saved_models/v2_feature_tushare/model-{config.ds}-x5_seed{config.seed}.pkl"
    if config.only_pos:
        relation_num = RELATION_NUM['MAX'] + 1
    else:
        relation_num = 2 * RELATION_NUM["MAX"] + 1

    net = HGTModule(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim,
                      relation_num=relation_num, node_num=1,
                      batch_size=config.batch_size, dropout=config.dropout, num_heads=config.num_heads,
                      num_layers=config.num_gnn_layer, agg_mode=config.agg_mode, pooling_type=config.pooling_type)
    net.load_state_dict(torch.load(model_save_path, map_location=device))
    print("load model")
    return net

@app.route('/predict', methods=["POST"])
def predict():
    if not request.method == "POST":
        return
    data = request.values.getlist('data')
    cols = request.values.getlist('cols')
    split_data = []
    for i in range(5):
        split_data.extend([data[i * len(cols):(i + 1) * len(cols)]])
    df = pd.DataFrame(split_data, columns=cols)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore', downcast='float')

    graph = get_graph(df)
    graph.edata["edge_feature"] = graph.edata["edge_feature"] + RELATION_NUM['MAX']
    graph.ndata["u_node_type"] = graph.ndata.pop("node_type")
    node_fea = graph.ndata.pop('node_fea')
    node_masks = graph.ndata.pop('mask_idx')
    with torch.no_grad():
        graph = graph.to(device)
        node_fea = node_fea.to(device)
        node_masks = node_masks.to(device)
        logits = model.forward(graph, node_fea, node_masks)
        print(logits)
        result = torch.argmax(logits, dim=-1).to(torch.int64).detach().cpu().tolist()

    response = {'result': result[0]}
    return jsonify(response)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Testing')
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model()
    model = model.to(device)
    model.eval()
    app.run(host="127.0.0.1", port=5001)