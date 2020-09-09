import os
import argparse

import pandas as pd
import torch
from numpy import array

from sklearn.metrics import accuracy_score


def read_csv(csv_path):
    return pd.read_csv(csv_path, header=None, index_col=False)


def evaluate(query, gallery, pred):
    
    assert query.shape[0] == pred.shape[0]
    
    pred = pred.squeeze()
    qurey_id = query[:, 0].tolist()
    pred_id  = []
    gallery_dic = dict(zip(gallery[:,1], gallery[:,0]))
    
    for p in pred:
        pred_id.append(gallery_dic[p])
    
    return accuracy_score(qurey_id, pred_id)*100


def evaluate_top5(query, gallery, pred):
    assert query.shape[0] == pred.shape[0]

    pred = pred.squeeze()
    qurey_id = query[:, 0].tolist()
    pred_id = []
    gallery_dic = dict(zip(gallery[:, 1], gallery[:, 0]))

    for idx, ps in enumerate(pred):
        pred_id.append([])
        for p in ps:
            pred_id[idx].append(gallery_dic[p])
    total = 0
    correct = 0
    for i in range(len(qurey_id)):
        total += 1
        if qurey_id[i] in pred_id[i]:
            correct += 1
    return (correct/total) * 100

def get_acc(model, query, gallery):
    with torch.no_grad():
        q_arr = []
        q_name = 0
        loss_c = torch.nn.CosineSimilarity(dim = 0)
        loss = torch.nn.MSELoss()
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
            min_e = 9999
            pred = ''
            max_e = 0
            pred_c = ''
            for g, n in zip(g_out, q_name):
                error = loss(q, g)
                error_c = loss_c(q, g)
                # > for cosine, < for MSE
                if error < min_e:
                    min_e = error
                    pred = n
                if error_c > max_e:
                    max_e = error_c
                    pred_c = n
            preds.append(pred)
            preds_c.append(pred_c)
    q_csv = read_csv('dataset/query.csv')
    g_csv = read_csv('dataset/gallery.csv')
    acc = evaluate(q_csv.values, g_csv.values, array(preds))
    acc_c = evaluate(q_csv.values, g_csv.values, array(preds_c))

    return acc, acc_c

def get_output(model, query, gallery):
    with torch.no_grad():
        q_arr = []
        q_name = 0
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
    #     for q in q_arr:
    #         # min_e has to be 0 for cosine similarity and a big number for MSELoss
    #         min_e = 9999
    #         pred = ''
    #         max_e = 0
    #         pred_c = ''
    #         for g, n in zip(g_out, q_name):
    #             error = loss(q, g)
    #             error_c = loss_c(q, g)
    #             # > for cosine, < for MSE
    #             if error < min_e:
    #                 min_e = error
    #                 pred = n
    #             if error_c > max_e:
    #                 max_e = error_c
    #                 pred_c = n
    #         preds.append(pred)
    #         preds_c.append(pred_c)
    #
    # acc = evaluate(q_csv.values, g_csv.values, array(preds))
    # acc_c = evaluate(q_csv.values, g_csv.values, array(preds_c))

    return q_arr



if __name__ == '__main__':

    '''argument parser'''
    parser = argparse.ArgumentParser(description='Code to evaluate DLCV final challenge 1')
    parser.add_argument('--query', type=str, help='path to query.csv') 
    parser.add_argument('--gallery', type=str, help='path to gallery.csv')
    parser.add_argument('--pred', type=str, help='path to your predicted csv file (e.g. predict.csv)')
    args = parser.parse_args()

    ''' read csv files '''
    query = read_csv(args.query)
    gallery = read_csv(args.gallery)
    pred = read_csv(args.pred)

    rank1 = evaluate(query.values, gallery.values, pred.values)
    
    print('===> rank1: {}%'.format(rank1))
