"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from torch.distributions.multivariate_normal import MultivariateNormal

from wols import RanPAC, AdaBoostClassifier, SAMME

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                logits = output['logits']

                if args.train_mask and class_mask is not None:
                    mask = []
                    for id in range(task_id + 1):
                        mask.extend(class_mask[id])
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    prompt_id = torch.max(logits, dim=1)[1]
                    # translate cls to task_id
                    prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                        -1)
                else:
                    prompt_id = None
            else:
                raise NotImplementedError("original model is None")
        output = model(input, task_id=task_id, prompt_id=prompt_id, train=set_training_mode,
                       prompt_momentum=args.prompt_momentum)
        logits = output['logits']
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target)  # base criterion (CrossEntropyLoss)
        # TODO add contrastive loss
        loss += orth_loss(output['pre_logits'], target, device, args)
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    prompt_id = torch.max(logits, dim=1)[1]
                    # translate cls to task_id
                    prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                        -1)
                    # print(prompt_id)
                else:
                    raise NotImplementedError("original model is None")

            output = model(input, task_id=task_id, prompt_id=prompt_id)
            logits = output['logits']
            promtp_idx = output['prompt_idx']  # tensor B x topk

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[i]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            task_inference_acc = utils.task_inference_accuracy(promtp_idx, target, target_task_map)

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

# @torch.no_grad()
def train_ranpac(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((4, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
    
    if args.num_view > 1:
        train_ranpac_multiView(model, original_model, data_loader,
                      device, task_id, class_mask, target_task_map, acc_matrix, args, )
        return
    
    print('TRAIN RANPAC.................................')
    device = torch.device(args.device)
    model_RanPAC = RanPAC(768, 10000, 200)

    for i in range(task_id + 1):
        data_loader_i=data_loader[i]['train']
        if args.distributed and utils.get_world_size() > 1:
            data_loader_i.sampler.set_epoch(1)
        Features_f = []
        label_list = []
        
        for input, target in data_loader_i:# metric_logger.log_every(data_loader_i, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                # find prompt_id
                if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        prompt_id = torch.max(logits, dim=1)[1]
                        # translate cls to task_id
                        prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                            -1)
                else:
                        raise NotImplementedError("original model is None")

                embedding = model.forward_features(input, task_id=task_id, prompt_id=prompt_id, prompt_weight=None, train=False, prompt_momentum=args.prompt_momentum)['x'][:, 0]
                Features_f.append(embedding.detach().cpu())
                label_list.append(target.cpu())
                
        Features_f = torch.cat(Features_f, dim=0).detach()
        label_list = torch.cat(label_list, dim=0).detach()
            
        print("Computing W...")
        print(Features_f.shape)
        model_RanPAC.compute_W(Features_f, label_list)
            
            
    print('TEST RANPAC .....................................')
    for i in range(task_id + 1):
        data_loader_i=data_loader[i]['val']
        if args.distributed and utils.get_world_size() > 1:
            data_loader_i.sampler.set_epoch(1)
        total = 0
        correct = 0
        with torch.no_grad():
            for input, target in data_loader_i: # metric_logger.log_every(data_loader_i, args.print_freq, header):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # find prompt_id
                if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        prompt_id = torch.max(logits, dim=1)[1]
                        # translate cls to task_id
                        prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                            -1)
                else:
                        raise NotImplementedError("original model is None")

                embedding = model.forward_features(input, task_id=task_id, prompt_id=prompt_id, prompt_weight=None, train=False, prompt_momentum=args.prompt_momentum)['x'][:,0]
                features = embedding.cpu().detach()
                features_h = torch.nn.functional.relu(features@ model_RanPAC.rp.cpu())
                output = features_h.mm(model_RanPAC.W0.cpu().t())
                predict = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].cpu().view(-1)
                
                total += len(target)
                correct += (predict.cpu() == target.cpu()).sum()
                
            print('Task', i, '- Acc: ', correct/total)

    return 

def train_ranpac_multiView(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    
    print('TRAIN RANPAC MULTIPLE VIEWS.................................')
    device = torch.device(args.device)
    
    base_model = RanPAC(768, 10000, args.nb_classes)
    MoE = AdaBoostClassifier(base_estimator=base_model,
                                n_estimators=args.num_view,
                                learning_rate=1)

    for i in range(task_id + 1):
        data_loader_i=data_loader[i]['train']
        if args.distributed and utils.get_world_size() > 1:
            data_loader_i.sampler.set_epoch(1)
            
        Features_f = []
        label_list = []
        
        for input, target in data_loader_i:# metric_logger.log_every(data_loader_i, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                # find prompt_id
                if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        prompt_id = torch.max(logits, dim=1)[1]
                        # translate cls to task_id
                        prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                            -1)
                else:
                        raise NotImplementedError("original model is None")

                embedding = model.forward_features(input, task_id=task_id, prompt_id=prompt_id, prompt_weight=None, train=False, prompt_momentum=args.prompt_momentum)['x'][:, 0]
                Features_f.append(embedding.detach().cpu())
                label_list.append(target.cpu())
                
        Features_f = torch.cat(Features_f, dim=0).detach().cpu()
        label_list = torch.cat(label_list, dim=0).detach().cpu()
            
        print("Learning...")
        # print(Features_f.shape)
        MoE.fit(Features_f, label_list)
            
    print('TEST RANPAC .....................................')
    for i in range(task_id + 1):
        data_loader_i=data_loader[i]['val']
        if args.distributed and utils.get_world_size() > 1:
            data_loader_i.sampler.set_epoch(1)
        total = 0
        correct = 0
        with torch.no_grad():
            for input, target in data_loader_i: # metric_logger.log_every(data_loader_i, args.print_freq, header):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # find prompt_id
                if original_model is not None:
                        output = original_model(input)
                        logits = output['logits']
                        if args.train_mask and class_mask is not None:
                            mask = []
                            for id in range(task_id + 1):
                                mask.extend(class_mask[id])
                            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        prompt_id = torch.max(logits, dim=1)[1]
                        # translate cls to task_id
                        prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device).unsqueeze(
                            -1)
                else:
                        raise NotImplementedError("original model is None")

                embedding = model.forward_features(input, task_id=task_id, prompt_id=prompt_id, prompt_weight=None, train=False, prompt_momentum=args.prompt_momentum)['x'][:,0]
                features = embedding.cpu().detach()
                features_h = torch.nn.functional.relu(features@ model_RanPAC.rp.cpu())
                
                predict = MoE.predict(features_h)
                # output = features_h.mm(model_RanPAC.W0.cpu().t())
                # predict = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].cpu().view(-1)
                
                total += len(target)
                correct += (predict.cpu() == target.cpu()).sum()
                
            print('Task', i, '- Acc: ', correct/total)

    return 


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()
    
    for task_id in range(6):
        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)

    for task_id in range(6, args.num_tasks):
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            if args.larger_prompt_lr:
                # This is a simple yet effective trick that helps to learn task-specific prompt better.
                base_params = [p for name, p in model_without_ddp.named_parameters() if
                            'prompt' in name and p.requires_grad == True]
                base_fc_params = [p for name, p in model_without_ddp.named_parameters() if
                                'prompt' not in name and p.requires_grad == True]
                base_params = {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
                base_fc_params = {'params': base_fc_params, 'lr': args.lr * 0.1, 'weight_decay': args.weight_decay}
                network_params = [base_params, base_fc_params]
                optimizer = create_optimizer(args, network_params)
            else:
                optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # load original model checkpoint
        if args.trained_original_model:
            original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = torch.load(original_checkpoint_path, map_location=device)
                original_model.load_state_dict(original_checkpoint['model'])
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
        # if model already trained
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
        
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (
                        slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(cur_start, cur_end))
                    prev_idx = (
                        slice(None), slice(None),
                        slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (
                        slice(None), slice(prev_start, prev_end))

                    try:
                    
                        with torch.no_grad():
                            if args.distributed:
                                model.module.e_prompt.prompt.grad.zero_()
                                model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                                # optimizer.param_groups[0]['params'] = model.module.parameters()
                            else:
                                model.e_prompt.prompt.grad.zero_()
                                model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                                # optimizer.param_groups[0]['params'] = model.parameters()
                    except:
                        pass

        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                            device=device, epoch=epoch, max_norm=args.clip_grad,
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        if args.prompt_momentum > 0 and task_id > 0:
            if args.use_prefix_tune_for_e_prompt:
                with torch.no_grad():
                    print(model.module.e_prompt.prompt[:, :, task_id].shape)
                    print(
                        model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(dim=2, keepdim=True).shape)
                    model.module.e_prompt.prompt[:, :, task_id].copy_(
                        (1 - args.prompt_momentum) * model.module.e_prompt.prompt[:, :, task_id].detach().clone()
                        + args.prompt_momentum * model.module.e_prompt.prompt[:, :, 0:task_id].detach().clone().mean(
                            dim=2))

        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)

        if task_id > 0 and not args.not_train_ca:
            pre_ca_test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                                  device=device,
                                                  task_id=task_id, class_mask=class_mask,
                                                  target_task_map=target_task_map,
                                                  acc_matrix=pre_ca_acc_matrix, args=args)

            train_task_adaptive_prediction(model, args, device, class_mask, task_id)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                   '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                      'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                

@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        dist.barrier()
        dist.all_gather(features_per_cls_list, features_per_cls)

        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' not in n]
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    cov = cls_cov[c_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)

                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
               for c_id in class_mask[i]:
                   for cluster in range(len(cls_mean[c_id])):
                       mean = cls_mean[c_id][cluster]
                       var = cls_cov[c_id][cluster]
                       if var.mean() == 0:
                           continue
                       m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                       sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                       sampled_data.append(sampled_data_single)
                       sampled_label.extend([c_id] * num_sampled_pcls)
        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)
        print(sampled_data.shape)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        # print(targets)

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id + 1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            #for name, p in model.named_parameters():
            #    if p.requires_grad and p.grad is None:
            #        print(name)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()


def orth_loss(features, targets, device, args):
    if cls_mean:
        # orth loss of this batch
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)
        M = torch.cat([sample_mean, features], dim=0)
        sim = torch.matmul(M, M.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        # print(loss)
        return args.reg * loss
    else:
        sim = torch.matmul(features, features.t()) / 0.8
        loss = torch.nn.functional.cross_entropy(sim, torch.range(0, sim.shape[0] - 1).long().to(device))
        return args.reg * loss
        # return 0.

