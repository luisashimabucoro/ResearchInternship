# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch
from torch import nn
import math
import time

from core.utils import accuracy, get_shuffle, check_gpu_memory_usage, create_split_list

class AbstractEvaluator(object):
    """
    Init:
        - type of model
        - episode size, way, shot etc
        - type of evaluation (cross-val, regular etc)
    
    Basic methods:
        - evaluate episode
        - evaluate batch

    """
    def __init__(self, model, **kwargs):
        super(AbstractEvaluator, self).__init__()

        self.model = model

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    
    def eval_episode(self, eval_type, episode_size, support_feat, support_target, query_feat, query_target, k_fold=None):
        if eval_type == 'oracle':
            outputs = self.oracle_evaluation(episode_size, support_feat, support_target, query_feat, query_target)
        elif eval_type == 'cross_val':
            outputs = self.cross_val_evaluation(episode_size, support_feat, support_target, query_feat, k_fold)
        else:
            print(f"Evaluation method does not exist.")
            outputs = None

        return outputs
    
    
    def eval_batch(self, episode_size, support_feat, support_target, query_feat, query_target, k_fold=None):
        eval_results = {}

        for eval_type in self.model.eval_types:
            outputs = self.eval_episode(eval_type, episode_size, support_feat, support_target, query_feat, query_target, k_fold)
            eval_results[eval_type] = outputs
        
        return eval_results
    
    @abstractmethod
    def oracle_evaluation(self, *args, **kwargs):
        return
    
    @abstractmethod
    def cross_val_evaluation(self, *args, **kwargs):
        return



class BaselineEvaluator(AbstractEvaluator):
    def __init__(self, model, **kwargs):
        super(BaselineEvaluator, self).__init__(model, **kwargs)

    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        output_list = []
        for episode_idx in range(episode_size):
            _, output = self.model.set_forward_adaptation(support_feat[episode_idx], support_target[episode_idx], query_feat[episode_idx])
            output_list.append(torch.from_numpy(output))

        # print(output_list)
        output = torch.cat(output_list, dim=-1).to(self.model.device)
        # print(f"Output size: {output.size()} and values {output}")
        # print(f"Query target size: {query_target.size()} and values {query_target}")
        # number_of_classes: 5
        # batch_size: 15
        # output:(number_of_classes * batch_size)
        # query_target: (number_of_classes * batch_size)
        loss = self.model.loss_func(output, torch.squeeze(query_target))
        acc = accuracy(output, query_target.reshape(-1))

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}
    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold):
        cv_loss = []
        cv_acc = []
        output_list = []
        # print(self.model.test_way)
        # print(self.model.test_shot)

        for episode_idx in range(episode_size):
            # evenly spacing support set
            # eg: 0,  5, 10, 15, 20,  1,  6, 11, 16, 21,  2,  7, 12, 17, 22,  3,  8, 13, 18, 23,  4,  9, 14, 19, 24
            shuffle_idx = get_shuffle(self.model.test_way, self.model.test_shot)
            shuffled_feat = support_feat[episode_idx][shuffle_idx].clone()
            shuffled_target = support_target[episode_idx][shuffle_idx].clone()
            # creating folds
            split_size_list = create_split_list(self.model.test_way * self.model.test_shot, k_fold)
            shuffled_feat = list(torch.split(shuffled_feat, split_size_list))
            shuffled_target = list(torch.split(shuffled_target, split_size_list))

            for idx in range(k_fold):
                # dividing train folds from test fold
                feat_folds = shuffled_feat.copy()
                target_folds = shuffled_target.copy()
                cv_support_feat_test = feat_folds.pop(idx)
                cv_support_target_test = target_folds.pop(idx)
                cv_support_feat = torch.cat(feat_folds)
                cv_support_target = torch.cat(target_folds)
                
                classifier, _ = self.model.set_forward_adaptation(cv_support_feat, cv_support_target, query_feat[0])
                output = classifier.predict_proba(cv_support_feat_test.cpu())
                output = torch.from_numpy(output).to(self.model.device)

                experimental_loss = self.model.loss_func(output, cv_support_target_test)
                # print(output.size())
                # print(output)
                # print(cv_support_target_test.size())
                # print(cv_support_target_test)
                acc = accuracy(output, cv_support_target_test)

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)
                # print(f"Fold {idx+1} completed!")
        
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc}





class ProtoEvaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(ProtoEvaluator, self).__init__(model, **kwargs)

    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        output = self.model.proto_layer(query_feat, support_feat, self.model.way_num, self.model.shot_num, self.model.query_num
                                        ).reshape(episode_size * self.model.way_num * self.model.query_num, self.model.way_num)
        
        loss = self.model.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}

    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold):
        # return
        cv_loss = []
        cv_acc = []
        output_list = []

        episode_shuffle_idx = get_shuffle(self.model.test_way, self.model.test_shot)
        split_size_list = create_split_list(self.model.test_way * self.model.test_shot, k_fold)
        episode_shuffle_idx = list(torch.split(episode_shuffle_idx, split_size_list))

        for episode_idx in range(episode_size):
            cv_support_feat = support_feat[episode_idx].clone()
            cv_support_target = support_target[episode_idx].clone()
            
            for idx in range(k_fold):
                shuffle_idx = episode_shuffle_idx.copy()
                test_idx = shuffle_idx.pop(idx)
                train_idx, _ = torch.sort(torch.cat(shuffle_idx))
                cv_support_feat = support_feat[episode_idx].clone()
                cv_support_target = support_target[episode_idx].clone()
                cv_support_feat_test = cv_support_feat[test_idx]
                cv_support_target_test = cv_support_target[test_idx]
                cv_support_feat = cv_support_feat[train_idx]
                cv_support_target = cv_support_target[train_idx]

                # defining number of samples per class used to determine the class prototypes
                train_samples_per_class = {class_num : self.model.test_shot for class_num in range(self.model.test_way)}
                for idx in cv_support_target_test:
                    train_samples_per_class[int(idx)] -= 1
                
                # (batch, n_samples, feat_size)
                output = self.model.proto_layer(
                    cv_support_feat_test[None, ...], cv_support_feat[None, ...], self.model.way_num, self.model.shot_num, self.model.query_num, experimental=True, samples_per_class=train_samples_per_class
                )
                # print(k_fold)
                # print(output.size())
                if output.size() == torch.Size([1, 1, self.model.test_way]):
                    output = torch.squeeze(output, 0)
                else: 
                    output = torch.squeeze(output)

                # experimental_loss = self.model.loss_func(output, cv_support_target_test[None, ...])
                experimental_loss = self.model.loss_func(output, cv_support_target_test)
                acc = accuracy(output, cv_support_target_test.reshape(-1))

                cv_loss.append(experimental_loss)
                cv_acc.append(acc)
                output_list.append(output)

        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        return {'output' : output, 'loss' : cv_loss, 'accuracy' : cv_acc}
    




class MAMLEvaluator(AbstractEvaluator):

    def __init__(self, model, **kwargs):
        super(MAMLEvaluator, self).__init__(model, **kwargs)
    
    def oracle_evaluation(self, episode_size, support_feat, support_target, query_feat, query_target):
        # (ep_size, batch_size, channels, height, width) ?
        _, _, c, h, w = support_feat.size()

        output_list = []
        # iterating through episodes
        for i in range(episode_size):
            episode_support_image = support_feat[i].contiguous().reshape(-1, c, h, w)
            episode_support_target = support_target[i].reshape(-1)
            episode_query_image = query_feat[i].contiguous().reshape(-1, c, h, w)
            # episode_query_target = query_target[i].reshape(-1)

            # fine-tuning model with support set
            loss = self.model.set_forward_adaptation(episode_support_image, episode_support_target)

            # testing model on the query set
            output = self.model.forward_output(episode_query_image)

            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.contiguous().view(-1))
        print(f"Oracle Stats - Loss: {loss}\t Acc: {acc}")

        return {'output' : output, 'loss' : loss, 'accuracy' : acc}
    
    def cross_val_evaluation(self, episode_size, support_feat, support_target, query_feat, k_fold):
        episode_shuffle_idx = get_shuffle(self.model.test_way, self.model.test_shot)
        split_size_list = create_split_list(self.model.test_way * self.model.test_shot, k_fold)
        episode_shuffle_idx = list(torch.split(episode_shuffle_idx, split_size_list))

        _, _, c, h, w = support_feat.size()

        cv_loss = []
        cv_acc = []
        output_list = []
        # iterating through episodes
        for episode_idx in range(episode_size):
            cv_support_feat = support_feat[episode_idx].clone()
            cv_support_target = support_target[episode_idx].clone()

            # (ep_size, batch_size, channels, height, width)
            # (support_samples, channels, height, width)
            for idx in range(k_fold):
                shuffle_idx = episode_shuffle_idx.copy()
                test_idx = shuffle_idx.pop(idx)
                train_idx = torch.cat(shuffle_idx)

                cv_support_feat = support_feat[episode_idx].contiguous().reshape(-1, c, h, w)
                cv_support_target = support_target[episode_idx].reshape(-1)
                cv_support_feat_test = cv_support_feat[test_idx]
                cv_support_target_test = cv_support_target[test_idx]
                cv_support_feat = cv_support_feat[train_idx]
                cv_support_target = cv_support_target[train_idx]

                # fine-tuning model with support set
                loss = self.model.set_forward_adaptation(cv_support_feat.contiguous(), cv_support_target.reshape(-1))

                # testing model on the query set
                training_shuffle_idx = torch.randperm(cv_support_feat_test.size(0))
                output = self.model.forward_output(cv_support_feat_test[training_shuffle_idx].contiguous())
                acc = accuracy(output, cv_support_target_test[training_shuffle_idx].contiguous().view(-1))
                output_list.append(output)
                cv_loss.append(loss)
                cv_acc.append(acc)
                
        cv_loss = sum(cv_loss) / len(cv_loss)
        cv_acc = sum(cv_acc) / len(cv_acc)
        output = torch.cat(output_list, dim=0)

        print(f"CV Stats - Loss: {cv_loss}\t Acc: {cv_acc}")

        return {'output' : output_list, 'loss' : cv_loss, 'accuracy' : cv_acc}