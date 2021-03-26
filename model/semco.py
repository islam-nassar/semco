'''
Object oriented implementation for SSEM solution
Author: Islam Nassar
Date: 29-Jul-2020
'''
from __future__ import print_function
import random

import time
import os
import sys
import logging
import numpy as np
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from model.models import ResNet50WithEmbeddingHead, WideResnetWithEmbeddingHead, \
    ResNet18WithEmbeddingHead
from datasets.dataloaders import get_train_loaders, get_val_loader, get_test_loader
from model.lr_scheduler import WarmupCosineLrScheduler
from model.ema import EMA
from utils.utils import accuracy, interleave, de_interleave
from utils.utils import AverageMeter, get_dataset_name, time_str

from model.label_embedding_guessor import LabelEmbeddingGuessor
from utils.labels2wv import get_grouping, get_labels2wv_dict


class SemCo:

    def __init__(self, config, dataset_meta, device, L='dynamic', device_ids=None):
        self.config = config
        self.dataset_meta = dataset_meta
        if 'stats' not in dataset_meta:
            self.dataset_meta['stats'] = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # imagenet_stats
        self.device = device
        self.parallel = config.parallel
        self.device_ids = device_ids
        self.L = L
        self.label_emb_guessor, self.emb_dim = self._get_label_guessor()
        self.model = self._set_model(config.parallel, device_ids)
        self.optim = self._get_optimiser(self.config)
        if not self.config.no_amp:
            from apex import amp
            self.model, self.optim = amp.initialize(self.model, self.optim, opt_level="O1")
        if self.config.parallel:
            self.model = nn.DataParallel(self.model)
        # initialise the exponential moving average model
        self.ema = EMA(self.model, self.config.ema_alpha)
        if self.config.use_pretrained:
            self.load_model_state(config.checkpoint_path)
        if self.config.freeze_backbone:
            self._freeze_model_backbone()
        self.logger, self.writer, self.time_stamp = self._setup_default_logging()

    def train(self, labelled_data, valid_data=None, training_config=None, save_best_model=False):
        """
        SemCo training function.
        labelled_data: dictionary holding labelled data in the form {'train/img1.png' : 'classA', ...}. This is relative
        to the dataset directory

        valid_data: dictionary holding validation data in the form {'train/img1.png' : 'classA', ...}. This is relative
        to the dataset directory

        training_config: to override parser config entirely if needed.

        save_best_model: if true, the best model (model state, ema state, optimizer state, classes) will be saved under
        './saved_models' directory.

        """
        # to allow overriding training params for different runs of training
        if training_config is None:
            training_config = self.config
        else:
            self.config = training_config

        L = len(labelled_data)
        n_iters_per_epoch, n_iters_all = self._init_training(training_config, L)
        # define criterion for upper(semantic embedding) and lower(discrete label) paths
        crit_lower = lambda inp, targ: F.cross_entropy(inp, targ, reduction='none')
        crit_upper = lambda inp, targ: 1 - F.cosine_similarity(inp, targ)
        optim = self.optim

        if n_iters_all == 0:
            n_iters_all = 1 # to avoid division by zero if we choose to set epochs to zero to skip a round
        lr_schdlr = WarmupCosineLrScheduler(optim, max_iter=n_iters_all, warmup_iter=0)

        num_workers = 0 if self.device == 'cpu' else training_config.num_workers_per_gpu * torch.cuda.device_count() if self.parallel else 4

        dltrain_x, dltrain_u = self._get_train_loaders(labelled_data, n_iters_per_epoch, num_workers, pin_memory=True,
                                                       cache_imgs=training_config.cache_imgs)
        print(f'Num of Labeled Training Data: {len(dltrain_x.dataset)}\nNum of Unlabeled Training Data:{len(dltrain_u.dataset)}')
        if valid_data:
            dlvalid = self._get_val_loader(valid_data, num_workers, pin_memory=True, cache_imgs=training_config.cache_imgs)
            print(f'Num of Validation Data: {len(dlvalid.dataset)}')

        train_args = dict(n_iters=n_iters_per_epoch, optim=optim, crit_lower=crit_lower,
                          crit_upper=crit_upper, lr_schdlr=lr_schdlr, dltrain_x=dltrain_x,
                          dltrain_u=dltrain_u)
        best_acc = -1
        best_epoch = 0
        best_loss = 1e6
        early_stopping_counter = 0
        best_metric = best_acc if training_config.es_metric == 'accuracy' else best_loss


        self.logger.info('-----------start training--------------')
        epochs_iterator = range(training_config.n_epoches) if not self.config.no_progress_bar else \
            tqdm(range(training_config.n_epoches),desc='Epoch')  # so that it displays the bar per epoch not per iteration
        for epoch in epochs_iterator:
            # training starts here
            train_loss, loss_x, loss_u, mask_mean, \
            loss_emb_x, loss_emb_u, mask_emb, mask_combined = \
                self._train_one_epoch(epoch, **train_args)
            if valid_data:
                top1, top5, valid_loss, top1_emb, top5_emb, top1_combined = self._evaluate(dlvalid, crit_lower)

            if valid_data:
                self.writer.add_scalars('train/1.loss', {'train': train_loss,
                                                         'test': valid_loss}, epoch)

            else:
                self.writer.add_scalar('train/1.loss', train_loss, epoch)
            self.writer.add_scalar('train/2.train_loss_x', loss_x, epoch)
            self.writer.add_scalar('train/2.train_loss_emb_x', loss_emb_x, epoch)
            self.writer.add_scalar('train/3.train_loss_u', loss_u, epoch)
            self.writer.add_scalar('train/3.train_loss_emb_u', loss_emb_u, epoch)
            self.writer.add_scalar('train/5.mask_mean', mask_mean, epoch)
            self.writer.add_scalar('train/5.mask_emb_mean', mask_emb, epoch)
            self.writer.add_scalar('train/5.mask_combined_mean', mask_combined, epoch)
            if valid_data:
                self.writer.add_scalars('test/1.test_acc', {'top1': top1, 'top5': top5, 'top1_emb': top1_emb,
                                                            'top5_emb': top5_emb, 'top1_combined': top1_combined},
                                        epoch)

                best_current = top1 if training_config.es_metric == 'accuracy' else valid_loss
                # only start looking for best model after min_wait period has expired
                if epoch >= training_config.min_wait_before_es:
                    isworse = lambda best,current: best <= current if training_config.es_metric == 'accuracy' else best >= current
                    if isworse(best_metric, best_current):
                        best_metric = best_current
                        best_epoch = epoch
                        if training_config.early_stopping_epochs:
                            best_model_state = self.model.state_dict()
                            best_ema_state = {k:v.clone().detach() for k,v in self.ema.shadow.items()}
                            early_stopping_counter = 0
                        if save_best_model:
                            try:
                                self._save_checkpoint()
                            except Exception as e:
                                print(f'Failed to save checkpoint: {e}')
                    elif training_config.early_stopping_epochs:
                        early_stopping_counter +=1
                else:
                    print('Minimum wait period still not expired. Leaving best epoch and best metric to default values')

                self.logger.info(
                    "Epoch {}. Top1: {:.4f}. Top5: {:.4f}. Top1_emb: {:.4f}. Top5_emb: {:.4f}. Top1_comb: {:.4f}. best_metric: {:.4f} in epoch{}".
                        format(epoch, top1, top5, top1_emb, top5_emb, top1_combined, best_metric, best_epoch))
                # check if early stopping is to be activated
                if training_config.early_stopping_epochs and early_stopping_counter == training_config.early_stopping_epochs:
                    self.logger.info(f"Early stopping activated, loading best models and ending training. "
                                     f"{training_config.early_stopping_epochs} epochs with no improvement.")
                    self.model.load_state_dict(best_model_state)
                    self.ema.shadow = best_ema_state
                    break
            # this will only be activated in last epoch to decide whether best model should be loaded or not before ending training
            if epoch == training_config.n_epoches-1:
                self.logger.info(f"Break epoch is reached")
                # in case early stopping is configured, load best model before exiting (edge case for early stopping)
                if training_config.early_stopping_epochs and valid_data and epoch >= training_config.min_wait_before_es:
                    self.logger.info(f"Loading best model and ending training (since early stopping is set)")
                    self.model.load_state_dict(best_model_state)
                    self.ema.shadow = best_ema_state
        self.writer.close()

    def predict(self):
        num_work = 0 if self.device == 'cpu' else 4
        dataloader = self._get_test_loader(num_work, pin_memory=True, cache_imgs=self.config.cache_imgs)
        # using EMA params to evaluate performance
        self.ema.apply_shadow()
        self.ema.model.eval()
        self.ema.model.to(self.device)

        predictions = []
        with torch.no_grad():
            for ims in dataloader:
                ims = ims.to(self.device)
                logits, _, _ = self.ema.model(ims)
                probs = torch.softmax(logits, dim=1)
                scores, lbs_guess = torch.max(probs, dim=1)
                predictions.append(lbs_guess)
            predictions = torch.cat(predictions).cpu().detach().numpy()

        predictions = [self.dataset_meta['classes'][elem] for elem in predictions]
        filenames = [name.split('/')[-1] for name in dataloader.dataset.data]
        df = pd.DataFrame({'id': filenames, 'class': predictions})

        # note roll back model current params to continue training
        self.ema.restore()

        return df

    def load_model_state(self, chkpt_dict_path):
        '''
        Loads model state based on a checkpoint saved by SemCo _save_checkpoint() function.
        '''
        print("Loading Model State")
        checkpoint_dict = torch.load(chkpt_dict_path, map_location=self.device)
        if 'model_state_dict' in checkpoint_dict:
            state_dict = checkpoint_dict['model_state_dict']
        else:
            print('model_state_dict key is not present in checkpoint, loading pretrained model failed, using original initialization for model')
            return

        # handle state_dictionaries where keys has 'module' in them (if the model was wrapped in nn.DataParallel)
        if all(['module' in key for key in state_dict.keys()]):
            if all(['module' in key for key in self.model.state_dict()]):
                pass
            else:
                state_dict= {k.replace('module.',''):v for k,v in state_dict.items()}
                if 'ema_shadow' in checkpoint_dict:
                    checkpoint_dict['ema_shadow'] = {k.replace('module.',''):v for k,v in checkpoint_dict['ema_shadow'].items()}

        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f'Problem occurred during naive state_dict loading: {e}.\nTrying to only load common params')
            try:
                model_state= self.model.state_dict()
                pretrained_state = {k:v for k,v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
                unloaded_state = set(list(state_dict.keys())) - set(list(model_state.keys()))
                model_state.update(pretrained_state)
                self.model.load_state_dict(model_state)
                print(f'Success. Following params in  pretrained_state_dict were not loaded: {unloaded_state}')
            except Exception as e:
                print(f'Unable to load model state due to following error. Model will be initialised randomly. \n {e}')
        if 'ema_shadow' in checkpoint_dict:
            try:
                self.ema = EMA(self.model, self.config.ema_alpha)
                similar_params = {k:v for k,v in checkpoint_dict['ema_shadow'].items() if k in self.ema.shadow and v.size() == self.ema.shadow[k].size()}
                self.ema.shadow.update(similar_params)
                print(f'EMA shadow has been loaded successfully. {len(similar_params)} out of {len(self.ema.shadow)} params were loaded')
            except Exception as e:
                print(f'Unable to load EMA shadow. EMA will be reinitialised with current model params. {e}')
                self.ema = EMA(self.model, self.config.ema_alpha)
        else:
            print('EMA shadow is not found in checkpoint dictionary. EMA will be reinitialised with current model params.')
            self.ema = EMA(self.model, self.config.ema_alpha)
        try:
            if 'classes' in checkpoint_dict:
                classes = self.dataset_meta['classes']
                classes_model = checkpoint_dict['classes']
                if all([classes_model[i] == classes[i] for i in range(len(classes))]):
                    print(f'classes matched successfully')
                else:
                    print(
                        "Classes loaded don't match the classes used while training the model, output of softmax can't be trusted")
        except Exception as e:
            print("can't load classes file. Pls check and try again.")
        return

    def adapt(self, num_classes):
        '''
        To allow adapting the model to a different dataset with the same semantic classifier weights
        num_classes: number of classes in the target dataset
        return: None
        '''
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.adapt(num_classes)
        else:
            self.model.adapt(num_classes)
        self.model.to(self.device)
        self.ema = EMA(self.model, self.config.ema_alpha)

    def _evaluate(self, dataloader, criterion):

        # using EMA params to evaluate performance
        self.ema.apply_shadow()
        self.ema.model.eval()
        self.ema.model.to(self.device)

        loss_meter = AverageMeter()
        top1_meter = AverageMeter()
        top5_meter = AverageMeter()
        top1_emb_meter = AverageMeter()
        top5_emb_meter = AverageMeter()
        top1_combined_meter = AverageMeter()

        with torch.no_grad():
            for ims, lbs in dataloader:
                ims = ims.to(self.device)
                lbs = lbs.to(self.device)
                logits, logits_emb, _ = self.ema.model(ims)
                sim = F.cosine_similarity(logits_emb.unsqueeze(1), self.label_emb_guessor.embedding_matrix.unsqueeze(0),
                                          dim=-1)
                sim = sim * self.label_emb_guessor.sharpening_factor
                loss = criterion(logits, lbs).mean()
                scores_emb = torch.softmax(sim, -1)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))
                top1_emb, top5_emb = accuracy(scores_emb, lbs, (1, 5))
                scores_combined = torch.mean(torch.stack([scores_emb, scores]), dim=0)
                top1_combined, _ = accuracy(scores_combined, lbs, (1, 5))
                loss_meter.update(loss.item())
                top1_meter.update(top1.item())
                top5_meter.update(top5.item())
                top1_emb_meter.update(top1_emb.item())
                top5_emb_meter.update(top5_emb.item())
                top1_combined_meter.update(top1_combined.item())

        # note roll back model current params to continue training
        self.ema.restore()
        return top1_meter.avg, top5_meter.avg, loss_meter.avg, top1_emb_meter.avg, top5_emb_meter.avg, top1_combined_meter.avg

    def _set_model(self, parallel, device_ids):
        classes = self.dataset_meta['classes']
        n = len(classes)
        if self.config.model_backbone is not None:
            if self.config.model_backbone == 'wres':
                model = WideResnetWithEmbeddingHead(num_classes=n, k=self.config.wres_k, n=28, emb_dim=self.emb_dim)
            elif self.config.model_backbone == 'resnet18':
                model = ResNet18WithEmbeddingHead(num_classes=n, emb_dim=self.emb_dim,
                                                  pretrained=not self.config.no_imgnet_pretrained)
            elif self.config.model_backbone == 'resnet50':
                model = ResNet50WithEmbeddingHead(num_classes=n, emb_dim=self.emb_dim,
                                                  pretrained=not self.config.no_imgnet_pretrained)
        # if no backbone is passed in args, auto infer based on im size
        elif self.config.im_size <= 64:
            model = WideResnetWithEmbeddingHead(num_classes=n, k=self.config.wres_k, n=28, emb_dim=self.emb_dim)
        else:
            model = ResNet50WithEmbeddingHead(num_classes=n, emb_dim=self.emb_dim,
                                              pretrained=not self.config.no_imgnet_pretrained)

        model.to(self.device)

        return model

    def _freeze_model_backbone(self):
        for name, param in self.model.named_parameters():
            if 'fc_emb' in name or 'fc_classes' in name:
                param.requires_grad = True
                print(f'{name} parameter is unfrozen')
            else:
                param.requires_grad = False
        print('All remaining parameters are frozen.')

    def _train_one_epoch(self, epoch, n_iters, optim, crit_lower, crit_upper,
                         lr_schdlr, dltrain_x, dltrain_u):

        # note: _x denotes supervised and _u denotes unsupervised
        # note: when suffix '_emb' is appended to variable, it denotes same variable but for upper path

        # Renaming for consistency
        criteria_x = crit_lower
        criteria_u = crit_lower
        criteria_x_emb = crit_upper
        criteria_u_emb = crit_upper
        if not self.config.no_amp:
            from apex import amp

        self.model.train()
        loss_meter = AverageMeter()
        loss_x_meter = AverageMeter()
        loss_u_meter = AverageMeter()
        loss_emb_x_meter = AverageMeter()
        loss_emb_u_meter = AverageMeter()
        # the number of gradient-considered strong augmentation (logits above threshold) of unlabeled samples
        n_strong_aug_meter = AverageMeter()
        max_score = AverageMeter()
        max_score_emb = AverageMeter()
        mask_meter = AverageMeter()
        mask_emb_meter = AverageMeter()
        mask_combined_meter = AverageMeter()

        epoch_start = time.time()  # start time
        dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
        iterator = range(n_iters) if self.config.no_progress_bar else tqdm(range(n_iters), desc='Epoch {}'.format(epoch))
        for it in iterator:
            ims_x_weak, ims_x_strong, lbs_x = next(dl_x)
            ims_u_weak, ims_u_strong = next(dl_u)
            lbs_x = lbs_x.to(self.device)

            bt = ims_x_weak.size(0)
            mu = int(ims_u_weak.size(0) // bt)
            imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong], dim=0).to(self.device)
            imgs = interleave(imgs, 2 * mu + 1)
            logits, logits_emb, _ = self.model(imgs)
            del imgs
            logits = de_interleave(logits, 2 * mu + 1)
            logits_x = logits[:bt]
            logits_u_w, logits_u_s = torch.split(logits[bt:], bt * mu)
            del logits

            logits_emb = de_interleave(logits_emb, 2 * mu + 1)
            logits_emb__x = logits_emb[:bt]
            logits_emb_u_w, logits_emb_u_s = torch.split(logits_emb[bt:], bt * mu)
            del logits_emb

            # supervised loss for upper and lower paths
            loss_x = criteria_x(logits_x, lbs_x).mean()
            loss_x_emb = criteria_x_emb(logits_emb__x, self.label_emb_guessor.embedding_matrix[lbs_x]).mean()

            # guessing the labels for upper and lower paths
            with torch.no_grad():
                probs = torch.softmax(logits_u_w, dim=1)
                scores, lbs_u_guess = torch.max(probs, dim=1)
                mask = scores.ge(self.config.thr).float()
                # get label guesses and mask based on embedding predictions (upper path)
                lbs_emb_u_guess, mask_emb, scores_emb, lbs_guess_help = self.label_emb_guessor(logits_emb_u_w)

            # combining the losses via co-training (blind version)
            mask_combined = mask.bool() | mask_emb.bool()
            # each loss path will have two components (co-training implementation)
            loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean() + \
                     (criteria_u(logits_u_s, lbs_guess_help) * mask_emb).mean() * (self.config.lambda_emb) / 3

            loss_u_emb = (criteria_u_emb(logits_emb_u_s, lbs_emb_u_guess) * mask_emb).mean() + \
                         (criteria_u_emb(logits_emb_u_s,
                                         self.label_emb_guessor.embedding_matrix[lbs_u_guess]) * mask).mean()

            loss_lower = loss_x + self.config.lam_u * loss_u
            loss_upper = loss_x_emb + self.config.lam_u * loss_u_emb
            loss = loss_lower + self.config.lambda_emb * loss_upper

            optim.zero_grad()
            if not self.config.no_amp:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optim.step()
            self.ema.update_params()
            lr_schdlr.step()

            loss_meter.update(loss.item())
            loss_x_meter.update(loss_x.item())
            loss_u_meter.update(loss_u.item())
            mask_meter.update(mask.mean().item())
            n_strong_aug_meter.update(mask_emb.sum().item())
            max_score.update(scores.mean())
            max_score_emb.update(scores_emb.mean())
            loss_emb_x_meter.update(loss_x_emb.item())
            loss_emb_u_meter.update(loss_u_emb.item())
            mask_combined_meter.update(mask_combined.float().mean().item())
            mask_emb_meter.update(mask_emb.mean().item())

            if (it + 1) % 512 == 0:
                t = time.time() - epoch_start

                lr_log = [pg['lr'] for pg in optim.param_groups]
                lr_log = sum(lr_log) / len(lr_log)

                self.logger.info("epoch:{}, iter: {}. loss: {:.4f}. loss_u: {:.4f}. loss_x: {:.4f}. max_score:{:.4f}. "
                                 " Mask:{:.4f} loss_u_emb:{:.4f}. loss_x_emb:{:.4f}. mask_emb:{:.4f}. max_score_emb:{:.4f}. mask_emb_count:{:.4f}. mask_combined:{:.4f}. . LR: {:.4f}. Time: {:.2f}".format(
                    epoch, it + 1, loss_meter.avg, loss_u_meter.avg, loss_x_meter.avg, max_score.avg,
                    mask_meter.avg, loss_emb_u_meter.avg, loss_emb_x_meter.avg, mask_emb_meter.avg, max_score_emb.avg,
                    n_strong_aug_meter.avg, mask_combined_meter.avg, lr_log, t))

                epoch_start = time.time()

        self.ema.update_buffer()
        return loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg, mask_meter.avg, \
               loss_emb_x_meter.avg, loss_emb_u_meter.avg, mask_emb_meter.avg, mask_combined_meter.avg

    def _get_train_loaders(self, labelled_data, n_iters_per_epoch, num_workers, pin_memory, cache_imgs):
        mean, std = self.dataset_meta['stats']
        kwargs = dict(dataset_path=self.config.dataset_path, classes=self.dataset_meta['classes'],
                      labelled_data=labelled_data, batch_size=self.config.batch_size, mu=self.config.mu,
                      n_iters_per_epoch=n_iters_per_epoch, size=self.config.im_size, cropsize=self.config.cropsize,
                      mean=mean, std=std, num_workers=num_workers, pin_memory=pin_memory, cache_imgs=cache_imgs)

        return get_train_loaders(**kwargs)

    def _get_val_loader(self, valid_data, num_workers, pin_memory, cache_imgs):
        mean, std = self.dataset_meta['stats']
        kwargs = dict(dataset_path=self.config.dataset_path, classes=self.dataset_meta['classes'],
                      labelled_data=valid_data, batch_size=3 * self.config.batch_size,
                      size=self.config.im_size, cropsize=self.config.cropsize,
                      mean=mean, std=std, num_workers=num_workers, pin_memory=pin_memory, cache_imgs=cache_imgs)

        return get_val_loader(**kwargs)

    def _get_test_loader(self, num_workers, pin_memory, cache_imgs):
        mean, std = self.dataset_meta['stats']
        kwargs = dict(dataset_path=self.config.dataset_path, classes=self.dataset_meta['classes'],
                      batch_size=3 * self.config.batch_size, size=self.config.im_size,
                      cropsize=self.config.cropsize, mean=mean, std=std, num_workers=num_workers,
                      pin_memory=pin_memory, cache_imgs=cache_imgs)

        return get_test_loader(**kwargs)

    def _get_label_guessor(self):
        classes = self.dataset_meta['classes']
        class_2_embeddings_dict = get_labels2wv_dict(classes, self.config.word_vec_path)
        emb_dim = len(list(class_2_embeddings_dict.values())[0])
        if self.config.eps is None:
            eps = 0.15 if emb_dim < 100 else 0.2 if emb_dim < 256 else 0.28  # for label grouping clustering
        else:
            eps = self.config.eps
        label_group_idx, gr_mapping = get_grouping(class_2_embeddings_dict, eps=eps, return_mapping=True)
        label_guessor = LabelEmbeddingGuessor(classes, label_group_idx, class_2_embeddings_dict, self.config.thr_emb,
                                              self.device)
        return label_guessor, emb_dim

    def _setup_default_logging(self, default_level=logging.INFO):

        format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"
        dataset_name = get_dataset_name(self.config.dataset_path)
        output_dir = os.path.join(dataset_name, f'x{self.L}')
        os.makedirs(output_dir, exist_ok=True)

        writer = SummaryWriter(comment=f'{dataset_name}_{self.L}')

        logger = logging.getLogger('train')
        logger.setLevel(default_level)

        time_stamp = time_str()
        logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
            filename=os.path.join(output_dir, f'{time_stamp}_{self.L}_labelled_instances.log'),
            format=format,
            datefmt="%m/%d/%Y %H:%M:%S",
            level=default_level)
        # to avoid double printing when creating new instances of class
        if not logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(default_level)
            console_handler.setFormatter(logging.Formatter(format))
            logger.addHandler(console_handler)
        #
        logger.info(dict(self.config._get_kwargs()))
        if self.device != 'cpu':
            logger.info(f'Device used: {self.device}_{torch.cuda.get_device_name(self.device)}')
        logger.info(f'Model:  {self.model.module.__class__ if isinstance(self.model, torch.nn.DataParallel) else self.model.__class__}')
        logger.info(f'Num_labels: {self.L}')
        logger.info(f'Image_size: {self.config.im_size}')
        logger.info(f'Cropsize: {self.config.cropsize}')
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in self.model.parameters()) / 1e6))

        return logger, writer, time_stamp

    def _init_training(self, training_config, L):

        n_iters_per_epoch = training_config.n_imgs_per_epoch // training_config.batch_size
        n_iters_all = n_iters_per_epoch * training_config.n_epoches
        if training_config.seed > 0:
            torch.manual_seed(training_config.seed)
            random.seed(training_config.seed)
            np.random.seed(training_config.seed)

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num Epochs = {training_config.n_epoches}")
        self.logger.info(f"  Early Stopping Epochs Patience = "
                         f"{training_config.early_stopping_epochs if training_config.early_stopping_epochs else None}")
        self.logger.info(f"  Minimum Wait before ES = {training_config.min_wait_before_es} epochs")
        self.logger.info(f"  Batch size Labelled = {training_config.batch_size}")
        self.logger.info(f"  Total optimization steps = {n_iters_all}")

        return n_iters_per_epoch, n_iters_all

    def _get_optimiser(self, training_config):
        # set weight decay to zero for batch-norm layers
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if 'bn' in name:
                non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias
            else:
                wd_params.append(param)
        param_list = [{'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
        optim = torch.optim.SGD(param_list, lr=training_config.lr, weight_decay=training_config.weight_decay,
                                momentum=training_config.momentum, nesterov=True)

        return optim

    def _save_checkpoint(self):
        save_dir = 'saved_models' #os.path.abspath(os.path.join(self.config.checkpoint_path, os.pardir))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        dataset_name = get_dataset_name(self.config.dataset_path)
        model_name = self.model.module._get_name() if isinstance(self.model, torch.nn.DataParallel) else self.model._get_name()
        checkpoint = {'ema_shadow':self.ema.shadow,
                      'model_state_dict': self.model.state_dict(),
                      'classes': self.dataset_meta['classes']}
        fpath = f'{save_dir}/{model_name}_{dataset_name}_{self.time_stamp}_checkpoint_dict.pth'
        torch.save(checkpoint,fpath)
        self.logger.info(f'Model Saved in: {fpath}')
