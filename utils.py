import torch
import data
import data_ng


def loader(args):
    if args.grouping:
        if args.random_sampling:
            train_loader = torch.utils.data.DataLoader(data.DATA2(args, mode='train'),
                                                       batch_size=args.train_batch,
                                                       num_workers=args.workers,
                                                       shuffle=True)
            gallery_loader = torch.utils.data.DataLoader(data.DATA2(args, mode='gallery'),
                                                         batch_size=40,
                                                         num_workers=args.workers,
                                                         shuffle=False)
            query_loader = torch.utils.data.DataLoader(data.DATA2(args, mode='query'),
                                                       batch_size=args.test_batch,
                                                       num_workers=args.workers,
                                                       shuffle=False)

        else:
            train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                                       batch_size=args.train_batch,
                                                       num_workers=args.workers,
                                                       shuffle=True)
            gallery_loader = torch.utils.data.DataLoader(data.DATA(args, mode='gallery'),
                                                         batch_size=40,
                                                         num_workers=args.workers,
                                                         shuffle=False)
            query_loader = torch.utils.data.DataLoader(data.DATA(args, mode='query'),
                                                       batch_size=args.test_batch,
                                                       num_workers=args.workers,
                                                       shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(data_ng.DATA(args, mode='train'),
                                                   batch_size=args.train_batch,
                                                   num_workers=args.workers,
                                                   shuffle=True)
        gallery_loader = torch.utils.data.DataLoader(data_ng.DATA(args, mode='gallery'),
                                                     batch_size=40,
                                                     num_workers=args.workers,
                                                     shuffle=False)
        query_loader = torch.utils.data.DataLoader(data_ng.DATA(args, mode='query'),
                                                   batch_size=args.test_batch,
                                                   num_workers=args.workers,
                                                   shuffle=False)

    return train_loader, gallery_loader, query_loader


def group_imgs(imgs, cls):
    all_img = []
    all_labels = []

    for img_list, lab in zip(imgs, cls):
        for i in range(len(img_list)):
            all_img.append(img_list[i])
            all_labels.append(lab)

    all_img = torch.stack(all_img).cuda()
    all_labels = torch.stack(all_labels)
    return all_img, all_labels
