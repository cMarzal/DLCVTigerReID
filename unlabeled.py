import os
import pandas as pd
import torchvision.transforms as transforms
from operator import itemgetter
from torch.utils.data import Dataset
from PIL import Image
import torch
import random
import parser1
import data
import csv
from featureextractor import Model


if __name__ == '__main__':

    args = parser1.arg_parse()
    model = Model().cuda()
    model.eval()
    model.load_state_dict(torch.load('log/save_place/model1_cos_68.pth.tar'))

    unlabeled_loader = torch.utils.data.DataLoader(data.DATA_un(args),
                                               batch_size=args.test_batch,
                                               num_workers=args.workers,
                                               shuffle=False)
    if 1:
        with torch.no_grad():
            img_arr = []
            name_arr = []
            loss = torch.nn.CosineSimilarity(dim = 0).cuda()
            print('Passing images through model')
            for imgs, names in unlabeled_loader:
                imgs = imgs.cuda()
                g_out, _, _, _ = model(imgs)
                img_arr.append(g_out)
                name_arr = name_arr + list(names)
            img_arr = torch.cat(img_arr)
            preds = []
            errors = []
            print('Obtaining more similar images')
            for id, img in enumerate(img_arr):
                max_e = 0
                pred = ''
                for i, img2 in enumerate(img_arr):
                    if id != i:
                        error = loss(img, img2)
                        if error > max_e:
                            max_e = error
                            pred = i
                preds.append(pred)
                errors.append(max_e)
                if (id + 1) % 10 == 0:
                    print('Done with image number', id+1, '/', len(img_arr))
        torch.save(preds, 'log/preds.pt')
        torch.save(errors, 'log/errors.pt')
    else:
        preds = torch.load('log/preds.pt')
        errors = torch.load('log/errors.pt')
        name_arr = []
        for _, names in unlabeled_loader:
            name_arr = name_arr + list(names)

    print('Obtaining top 50% error')
    top_err = sorted([(x.item(),i) for (i,x) in enumerate(errors)], reverse=True )[:int(len(errors)/2)]
    lab = [ele for _, ele in top_err]
    top_preds = [(preds[i], i) for i in lab]
    name_ids = []
    label = 0
    print('Obtaining relationships')
    for i in range(len(top_preds)):
        if i == 0:
            name_ids.append([label, top_preds[i][1]])
            name_ids.append([label, top_preds[i][0]])
            label += 1
        else:
            if any(top_preds[i][1] == ii[1] for ii in name_ids):
                if not any(top_preds[i][0] == ii[1] for ii in name_ids):
                    item = [item[0] for item in name_ids if item[1] == top_preds[i][1]]
                    item, = item
                    name_ids.append([item, top_preds[i][0]])
            else:
                if not any(top_preds[i][0] == i2[1] for i2 in name_ids):
                    name_ids.append([label,top_preds[i][1]])
                    name_ids.append([label,top_preds[i][0]])
                    label += 1
                else:
                    item = [item[0] for item in name_ids if item[1] == top_preds[i][0]]
                    item, = item
                    name_ids.append([item, top_preds[i][1]])

    count_z = [0] * 1000
    ind_list = []
    for i in range(len(name_ids)):
        count_z[name_ids[i][0]] += 1
    for i in range(len(count_z)):
        if count_z[i] >= 6:
            ind_list.append(i)
    final_range = []
    for i in range(len(name_ids)):
        if name_ids[i][0] in ind_list:
            final_range.append(name_ids[i])

    image_name_final = [name_arr[i[1]] for i in final_range]
    for i in range(len(final_range)):
        final_range[i][0] = final_range[i][0]+197

    with open(r'dataset/train.csv', 'a') as f:
        writer = csv.writer(f)
        for i in range(len(final_range)):
            writer.writerow((final_range[i][0], image_name_final[i]))

    print(name_ids)

