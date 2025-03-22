import torch
import numpy as np
import torch.nn.functional as F

class RanPAC():
    def __init__(self, dim_in, M, num_classes):
        self.rp = torch.nn.Linear(dim_in, M, bias=False)
        self.rp = torch.randn(dim_in, M)
        
        self.Q=torch.zeros(M, num_classes)
        self.G=torch.zeros(M,M)
        self.num_classes = num_classes
        self.W0 = None
        self.sample_weights = np.empty(0)
        self.curr_sample_weight =  None
        
    def target2onehot(self, targets, n_classes):
            onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
            onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
            return onehot
        
    def compute_W(self, Features_f, label_list):
        
        if self.curr_sample_weight.any() == None:
            self.curr_sample_weight = torch.ones(Features_f.shape[0]) 
        
        Y=self.target2onehot(label_list,self.num_classes)
        if True:
            print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            Features_h=torch.nn.functional.relu(Features_f@ self.rp.cpu())
            # print(Features_h.shape, Features_f.shape)
            # print(self.Q.shape, Features_h.shape, Y.shape)
            temp = Features_h.T @ np.diag(self.curr_sample_weight)
            # print(temp.shape, type(temp), type(Features_h), type(Y))
            # print(temp)
            # print(Y)
            # print(Features_h)
            
            # print(Features_h.shape, np.diag(self.curr_sample_weight).shape, Y.shape, self.curr_sample_weight.shape)
            # kk = temp @ Y
            
            self.Q=self.Q+(Features_h.T @ np.diag(self.curr_sample_weight)) @ Y 
            self.G=self.G+(Features_h.T @ np.diag(self.curr_sample_weight)) @ Features_h
            ridge=self.optimise_ridge_parameter(Features_h,Y)
            Wo=torch.linalg.solve(self.G+ridge*torch.eye(self.G.size(dim=0)),self.Q).T #better nmerical stability than .inv
            # print(Wo.shape)
            self.W0 = Wo.to(device='cuda')#[0:self._network.fc.weight.shape[0],:].
            
            # print('Done')
            
    def fit(self, Features_f, label_list, sample_weight):
        # self.sample_weights = np.concatenation((self.sample_weights, sample_weight), dim=0)
        self.curr_sample_weight = torch.Tensor(sample_weight)
        # print(self.sample_weights.shape)
        return self.compute_W(Features_f, label_list)
    
    def predict(self, Features_f, onehot=True):
        Features_f = Features_f.cpu().detach()
        Features_h = torch.nn.functional.relu(Features_f@ self.rp.cpu())
        output = Features_h.mm(self.W0.cpu().t())
        
        if onehot:
            predict = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].cpu().view(-1)
            y_pred=predict #self.target2onehot(predict,self.num_classes)
            return y_pred
        else:
            # print(output.shape)
            output = F.softmax(output, dim=1).max(1)[0]
            # print(output.shape)
            # print(output)
            # exit()
            return output#F.softmax(output, dim=1)
            
    def optimise_ridge_parameter(self,Features,Y):
        ridges=10.0**np.arange(-8,9)
        num_val_samples=int(Features.shape[0]*0.8)
        losses=[]
        Q_val=Features[0:num_val_samples,:].T @ Y[0:num_val_samples,:]
        G_val=Features[0:num_val_samples,:].T @ Features[0:num_val_samples,:]
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T #better nmerical stability than .inv
            Y_train_pred=Features[num_val_samples::,:]@Wo.T
            losses.append(F.mse_loss(Y_train_pred,Y[num_val_samples::,:]))
        ridge=ridges[np.argmin(np.array(losses))]
        print('selected lambda = ',ridge)
        return ridge
    

import numpy as np

"""
@author: Jia Geng
@email: jxg570@miami.edu
"""

class SAMME:
    """
    SAMME - multi-class AdaBoost algorithm
    @ref:   Zhu, Ji & Rosset, Saharon & Zou, Hui & Hastie, Trevor. (2006). Multi-class AdaBoost. Statistics and its
            interface. 2. 10.4310/SII.2009.v2.n3.a8.
    """

    def __init__(self, num_learner: int, num_cats: int):
        """
        Constructor
        :param num_learner: number of weak learners that will be boosted together
        :param num_cats: number of categories
        """

        if num_cats < 2:
            raise Exception("Param num_cat should be at least 2 but was {}".format(num_cats))

        self.num_learner = num_learner
        self.num_cats = num_cats
        self.entry_weights = None
        self.learner_weights = None
        self.sorted_learners = None

    def train(self, train_data: list, learners: list):
        """
        Train the AdaBoost .
        The training data need to be in the format: [(X, label), ...]
        The learners need to be in the format: [obj1, obj2, ...]
        The learner object need to have: a predict method that can output the predicted class. obj.predict(X) -> cat: int
        :param train_data: training data
        :param learners: weak learners
        :return: void
        """

        print("\nStart training SAMME..")
        # initialize the weights for each data entry
        n, m = len(train_data), len(learners)
        self.entry_weights = np.full((n,), fill_value=1/n, dtype=np.float32)
        self.learner_weights = np.zeros((m,), dtype=np.float32)

        # sort the weak learners by error
        error = [0 for i in range(m)]
        for learner_idx, learner in enumerate(learners):
            for entry in train_data:
                X, label = entry[0], int(entry[1])
                predicted_cat = learner.predict(X)
                if predicted_cat != label:
                    error[learner_idx] += 1
        self.sorted_learners = [l for l, e in sorted(zip(learners, error), key=lambda pair: pair[1])]

        # boost
        for learner_idx, learner in enumerate(self.sorted_learners):
            # compute weighted error
            is_wrong = np.zeros((n,))
            for entry_idx, entry in enumerate(train_data):
                X, label = entry[0], int(entry[1])
                predicted_cat = learner.predict(X)
                if predicted_cat != label:
                    is_wrong[entry_idx] = 1
            weighted_learner_error = np.sum(is_wrong * self.entry_weights)/self.entry_weights.sum()

            # compute alpha, if the learner is not qualified, set to 0
            self.learner_weights[learner_idx] = max(0, np.log(1/(weighted_learner_error + 1e-6) - 1) + np.log(
                self.num_cats - 1))
            alpha_arr = np.full((n,), fill_value=self.learner_weights[learner_idx], dtype=np.float32)
            # update entry weights, prediction made by unqualified learner will not update the entry weights
            self.entry_weights = self.entry_weights * np.exp(alpha_arr * is_wrong)
            self.entry_weights = self.entry_weights/self.entry_weights.sum()

        # normalize the learner weights
        self.learner_weights = self.learner_weights/self.learner_weights.sum()
        print("Training completed.")

    def predict(self, X):
        """
        Predict using the boosted learner
        :param X:
        :return: predict class
        """

        pooled_prediction = np.zeros((self.num_cats,), dtype=np.float32)

        for learner_idx, learner in enumerate(self.sorted_learners):
            # encode the prediction in to balanced array
            predicted_cat = learner.predict(X)
            prediction = np.full((self.num_cats,), fill_value=-1/(self.num_cats-1), dtype=np.float32)
            prediction[predicted_cat] = 1
            pooled_prediction += prediction*self.learner_weights[learner_idx]

        return np.argmax(pooled_prediction)
    

# __author__ = 'Xin'
# From: https://github.com/jinxin0924/multi-adaboost/blob/master/multi_AdaBoost.py

import numpy as np
# from numpy.core.umath_tests import inner1d
from copy import deepcopy


class AdaBoostClassifier(object):
    '''
    Parameters
    -----------
    base_estimator: object
        The base model from which the boosted ensemble is built.

    n_estimators: integer, optional(default=50)
        The maximum number of estimators

    learning_rate: float, optional(default=1)

    algorithm: {'SAMME','SAMME.R'}, optional(default='SAMME.R')
        SAMME.R uses predicted probabilities to update wights, while SAMME uses class error rate

    random_state: int or None, optional(default=None)


    Attributes
    -------------
    estimators_: list of base estimators

    estimator_weights_: array of floats
        Weights for each base_estimator

    estimator_errors_: array of floats
        Classification error for each estimator in the boosted ensemble.

    Reference:
    1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)

    2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/
    scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)

    '''

    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 5 # 50
        learning_rate = 1
        algorithm = 'SAMME'
        random_state = None

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')

        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.random_state_ = random_state
        self.estimators_ = list()
        for _ in range(self.n_estimators_):
            temp_estimator = deepcopy(self.base_estimator_)
            self.estimators_.append(temp_estimator)
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        self.estimator_errors_ = np.ones(self.n_estimators_)
        
        print(self)

    def _samme_proba(self, estimator, n_classes, X):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        proba = estimator.predict_proba(X)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])


    def fit(self, X, y):
        self.n_samples = X.shape[0]
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort
        self.classes_ = np.array(sorted(list(set(y))))
        # print(self.classes_)
        self.n_classes_ = len(self.classes_)
        i = 0
        for iboost in range(self.n_estimators_):
            print('--------------', i)
            i = i+1
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) #/ self.n_samples
                
            sample_weight, estimator_weight, estimator_error = self.boost(X, y, iboost, sample_weight)

            # early stop
            if estimator_error == None:
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            if estimator_error <= 0:
                break
            
        print('Doneeeeeeeeeee')

        return self


    def boost(self, X, y, iboost, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(X, y, iboost, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(X, y, iboost, sample_weight)

    def real_boost(self, X, y, iboost, sample_weight):
        # estimator = deepcopy(self.base_estimator_)
        estimator = self.estimators_[iboost]
        if self.random_state_:
            estimator.set_params(random_state=1)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_pred = estimator.predict(X)
        incorrect = y_pred != y
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # if worse than random guess, stop boosting
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            print('!!! Worse than random guess')
            return None, None, None

        y_predict_proba = estimator.predict(X, onehot=False).numpy() #estimator.predict_proba(X)
        # print(y_predict_proba[:10])
        # repalce zero
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps
        
        # print('--', y_predict_proba[:10])

        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])

        # for sample weight update
        intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) *
                                                              np.dot(y_coding, np.log(
                                                                  y_predict_proba))))  #dot iterate for each row

        # update sample weight
        sample_weight *= np.exp(intermediate_variable)
        
        # print('new w', sample_weight)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator)
        # print('Num estimator so far: ', len(self.estimator_))
        return sample_weight, 1, estimator_error

    def discrete_boost(self, X, y, iboost, sample_weight):
        # estimator = deepcopy(self.base_estimator_)
        estimator = self.estimators_[iboost]
        if self.random_state_:
            estimator.set_params(random_state=1)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_pred = estimator.predict(X).numpy()
        y = y.numpy()
        incorrect = y_pred != y
        # print(y_pred.shape, y_pred)
        # print(y.shape, y)
        # print(incorrect)
        # exit()
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
        # print(estimator)

        # if worse than random guess, stop boosting
        if estimator_error >= 1 - 1 / self.n_classes_:
            print('worse than random guess')
            return None, None, None

        # update estimator_weight
        estimator_weight = alpha = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(
            self.n_classes_ - 1)

        if estimator_weight <= 0:
            print('estimator_weight <= 0')
            return None, None, None

        # update sample weight
        sample_weight *= np.exp(estimator_weight * incorrect)
        
        print('new wD', sample_weight)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        sample_weight_mean = np.mean(sample_weight, axis=0) 
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_mean
        # print('new wD2', sample_weight)
        print('summ', sample_weight_sum)

        # append the estimator
        # self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)


    def predict_proba(self, X):
        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(self._samme_proba(estimator, self.n_classes_, X)
                        for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
    
    def voting(self, X):
        pass
