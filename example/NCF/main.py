# Code adapted from https://github.com/guoyang9/NCF

import os
import time
import argparse
import numpy as np

import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model
import evaluate
import data_utils
import os.path      

from FaaS.intra.elements import FaaSDataLoader
from FaaS.intra.job import IntraOptim
import FaaS.intra.env as env

def main():          

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.000001,
                        help="learning rate")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.1,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16384,
                        help="batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=20,
                        help="training epoches")
    parser.add_argument("--top_k",
                        type=int,
                        default=10,
                        help="compute metrics@top_k")
    parser.add_argument("--factor_num",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--num_layers",
                        type=int,
                        default=3,
                        help="number of layers in MLP model")
    parser.add_argument("--num_ng",
                        type=int,
                        default=4,
                        help="sample negative items for training")
    parser.add_argument("--test_num_ng",
                        type=int,
                        default=99,
                        help="sample part of negative items for testing")
    parser.add_argument("--out",
                        default=True,
                        help="save model or not")
    parser.add_argument("--gpu",
                        type=str,
                        default="1",
                        help="gpu card ID")
    parser.add_argument("--autoscale-bsz",
                        dest='autoscale_bsz',
                        default=False,
                        action='store_true',
                        help="Use AdaptDL batchsize autoscaling")
    parser.add_argument("--gradient-accumulation",
                        dest='gradient_accumulation',
                        default=False,
                        action='store_true',
                        help="Use AdaptDL batchsize autoscaling")
    parser.add_argument("--dataset",
                        type=str,
                        choices=['ml-1m', 'pinterest-20'],
                        default="ml-1m")
    parser.add_argument("--model-type",
                        dest="model_type",
                        type=str,
                        choices=['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre'],
                        default="NeuMF-end")
    parser.add_argument('--proxy_ip', default='127.0.0.1', type=str, help='ip of job-proxy')
    parser.add_argument('--proxy_port', default=12345, type=int, help='port of job-proxy')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # cudnn.benchmark = True


    dataset = args.dataset
    model_type = args.model_type

    # paths
    main_path = "./data"

    train_rating = os.path.join(main_path, '{}.train.rating'.format(dataset))
    test_rating = os.path.join(main_path, '{}.test.rating'.format(dataset))
    test_negative = os.path.join(main_path, '{}.test.negative'.format(dataset))

    model_path = os.path.join(main_path, 'models')
    GMF_model_path = os.path.join(model_path, 'GMF.pth')
    MLP_model_path = os.path.join(model_path, 'MLP.pth')
    NeuMF_model_path = os.path.join(model_path, 'NeuMF.pth')

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = \
        data_utils.load_all(main_path, train_rating, test_negative, dataset)

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
            train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(
            test_data, item_num, train_mat, 0, False)
    train_loader = FaaSDataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

    ########################### CREATE MODEL #################################
    if model_type == 'NeuMF-pre':
        assert os.path.exists(GMF_model_path), 'lack of GMF model'
        assert os.path.exists(MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(GMF_model_path)
        MLP_model = torch.load(MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    network = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                        args.dropout, model_type, GMF_model, MLP_model)
    device = env.local_rank()
    network = network.to(device)
    loss_function = torch.nn.BCEWithLogitsLoss()


    if model_type == 'NeuMF-pre':
        optimizer = optim.SGD(network.parameters(), lr=args.lr)
    else:
        #optimizer = optim.SGD(network.parameters(), lr=args.lr)
        optimizer = optim.Adam(network.parameters(), lr=args.lr)
    ########################### TRAINING #####################################
    
    job = IntraOptim(network, train_loader, optimizer, args.epochs, 1, args.proxy_ip, args.proxy_port)
    
    count, best_hr, epoch = 0, 0, 0
    while epoch < args.epochs:
        job.model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()
        job.beg_epoch()
        job.tiaoshi()
        epoch = job.get_epoch()
        for user, item, label in job.trainloader:
            with job.sync_or_not():
                user = user.to(device)
                item = item.to(device)
                label = label.float().to(device)
                job.model.zero_grad()
                prediction = job.model(user, item)
                loss = loss_function(prediction, label)
                loss.backward()
                job.optimizer.step()
                count += 1
                if job.is_break:
                    break
        job.model.eval()
        HR, NDCG = evaluate.metrics(job.model.module, test_loader, args.top_k)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
        
if __name__ == '__main__':
    main()
