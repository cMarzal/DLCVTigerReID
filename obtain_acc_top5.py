from evaluate import evaluate_top5
from featureextractor import Model
import parser1
import torch
import data
from numpy import array
import pandas as pd
import numpy as np


def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)


if __name__ == '__main__':

    args = parser1.arg_parse()
    model = Model().cuda()
    model.eval()
    model.load_state_dict(torch.load('log/model_best_Cos.pth.tar'))

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    gallery = torch.utils.data.DataLoader(data.DATA2(args, mode='gallery'),
                                                         batch_size=40,
                                                         num_workers=args.workers,
                                                         shuffle=False)

    query = torch.utils.data.DataLoader(data.DATA2(args, mode='query'),
                                                       batch_size=args.test_batch,
                                                       num_workers=args.workers,
                                                       shuffle=False)

    with torch.no_grad():
        q_arr = []
        q_name = 0
        loss_c = torch.nn.CosineSimilarity(dim = 0)
        loss = torch.nn.MSELoss()
        loss_class = torch.nn.CrossEntropyLoss()
        for g_img, g_name in gallery:
            g_img = g_img.cuda()
            q_name = g_name
            g_out, _, _, _ = model(g_img)
        for idx, (imgs, _) in enumerate(query):
            imgs = imgs.cuda()
            output, _, _, _ = model(imgs)
            q_arr.append(output)

        q_arr = torch.cat(q_arr)
        preds = []
        preds_c = []
        for q in q_arr:
            # min_e has to be 0 for cosine similarity and a big number for MSELoss
            min_e = [9999,9999,9999,9999,9999]
            pred = ['','','','','']
            max_e = [0,0,0,0,0]
            pred_c = ['','','','','']
            cnt = 0
            cnt_e = 0
            for g, n in zip(g_out, q_name):
                error = loss(q, g)
                error_c = loss_c(q, g)

                # > for cosine, < for MSE
                for x in range(5):
                    if error < min_e[x]:
                        min_e.insert(x, error)
                        pred.insert(x, n)
                        min_e.pop()
                        pred.pop()
                        break
                for x in range(5):
                    if error_c > max_e[x]:
                        max_e.insert(x, error_c)
                        pred_c.insert(x, n)
                        max_e.pop()
                        pred_c.pop()
                        break
            preds.append(pred)
            preds_c.append(pred_c)
    q_csv = read_csv('dataset/query.csv')
    g_csv = read_csv('dataset/gallery.csv')
    acc = evaluate_top5(q_csv.values, g_csv.values, array(preds))
    acc_c = evaluate_top5(q_csv.values, g_csv.values, array(preds_c))

    print('acc MSE:', acc)
    print('acc Cos:', acc_c)
