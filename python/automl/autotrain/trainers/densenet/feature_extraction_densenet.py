import math
import random
import warnings
import copy
from typing import Any, Union, Callable, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .configuration_densenet import DenseNetTrainerConfig
from ...utils.feature_extraction_utils import BaseFeatureExtractor, BaseFeatureExtractorOutput

@dataclass
class GADenseNetFeatureExtractorOutput(BaseFeatureExtractorOutput):
    final_data: Union[np.ndarray, List] = None
    best_feature_index: Union[np.ndarray, List]= None
    
@dataclass
class GABaseFeatureExtractorOutput(BaseFeatureExtractorOutput):
    final_data: Union[np.ndarray, List] = None
    best_feature_index: Union[np.ndarray, List] = None

class GAFeatureExtractor:
    def __init__(
        self, 
        config: DenseNetTrainerConfig
    ):
        """
        This is the implementation of Genetic Algorithm, for more information please refer
        :ref:`https://blog.csdn.net/littlely_ll/article/details/72625312`

        :param size: the population size

        :param feature_num: the number of features which to choose from

        :param svm_weight: SVM accuracy weight of fitness

        :param feature_weight: the features weight of fitness

        :param C: the features cost, which should be a number or numpy array, if numpy array, it must have the
                  same length with feature attributes

        :param keep_prob: the proportion of population

        :param cross_prob: the probability of gene cross

        :param mutate_prob: the probability of gene mutation

        :param iters: the iteration number of GA

        :param topK: select the best topK individuals to choose features

        :param topF: select feature when its count is more than topF
        """
        self.config = config
        # TODO inspect
        self.feature_num = config.dp_feature_num
        self.size = 10 * config.dp_feature_num
        self.svm_weight = config.dp_svm_weight
        self.feature_weight = config.dp_feature_weight
        self.C = config.dp_C
        self.keep_prob = config.dp_keep_prob
        self.cross_prob = 1 - config.dp_keep_prob
        self.mutate_prob = config.dp_mutate_prob
        self.iters = config.dp_iters
        self.topK = 2 * config.dp_feature_num
        self.topF = 0.8 * config.dp_feature_num

        self.average_fitness = []
        self.best_feature_index = []
        self.length = 10 * config.dp_feature_num
        
    def __call__(
        self, 
        inputs: Union[np.ndarray, pd.DataFrame, str],
        trainer: Callable = None,
        **kwargs: Any
    ) -> Any:
        if inputs is not None:
            if isinstance(inputs, str): # csv file path
                x_y = pd.read_csv(inputs)
                _, features_nums = x_y.shape
                x_train = x_y.iloc[:, :(features_nums - 1)].to_numpy()
                y_train = x_y.iloc[:, -1].to_numpy()
            elif isinstance(inputs, np.ndarray):
                raise NotImplementedError
            elif isinstance(inputs, pd.DataFrame):
                _, features_nums = inputs.shape
                x_train = inputs.iloc[:, :(features_nums - 1)].to_numpy()
                y_train = inputs.iloc[:, -1].to_numpy()
            else:
                raise ValueError("`inputs` must be np.ndarray, pd.DataFrame, or str")
        else:
            raise ValueError("You have to specify `inputs`")
        
        if not trainer:
             raise ValueError("You have to specify the trainer which belongs to Callable")
        self.trainer = trainer
        result_data, result_index = self.extract(x_train, y_train)
        
        return GABaseFeatureExtractorOutput(
            final_data=result_data,
            best_feature_index=result_index
        )
        
    def extract(self, X, y):
        """
        fit the array data

        :param X: the numpy array

        :param y: the label, a list or one dimension array

        :return:
        """

        fitness_array = np.array([])

        population = self.generate_population(size=self.size, feature_num=self.feature_num)
        for _iter in range(self.iters):
            print(f'!!!!!!!!!---------->{_iter / self.iters}')
            generators = self.features_and_params_generator(X, y, population=population, feature_num=self.feature_num)
            fitness_list = []

            for i, (_features, _y) in enumerate(generators):
                # 等推理接口好了以后，直接调用
                # pd.DataFrame(_features).to_csv(r'V:\project\automl\middle_data\ceshi_features.csv')
                # pd.DataFrame(_y).to_csv(r'V:\project\automl\middle_data\ceshi_y.csv')
                inputs = pd.DataFrame(pd.concat((pd.DataFrame(_features), pd.DataFrame(_y.reshape(-1, 1))), axis=1))
                trainer = copy.deepcopy(self.trainer)
                trainer.train(inputs=inputs)
                try:
                    trainer_summary = trainer.get_summary()
                    metric = trainer_summary.get('best_model_tracker', None).get('history', None).get('val_loss', None)[0]
                except:
                    raise ValueError('Unable to get trainer evaluation metrics')
                _fitness = self.fitness(population[i], metric, self.svm_weight, self.feature_weight, C=self.C)
                fitness_list.append(_fitness)
            fitness_array = np.array(fitness_list)

            if _iter != self.iters - 1:
                population = self.select_population(population, fitness_array, keep_prob=self.keep_prob)
                population = self.gene_cross(population, self.cross_prob)
                population = self.gene_mutate(population, self.mutate_prob)
                population = population[np.where(population[:, 0:self.feature_num].any(axis=1))[0], :]  # 至少有一个非零元素的行

        sorted_index = np.argsort(fitness_array)
        best_individuals = population[sorted_index[-self.topK:], :]

        feature_sum = best_individuals.sum(axis=0)
        for i in range(self.feature_num) :
            if feature_sum[i] >= self.topF :
                self.best_feature_index.append(i)

        return X[:, self.best_feature_index], self.best_feature_index


    @property
    def important_features(self):
        return self.best_feature_index

    def average_fitness(self):
        return self.average_fitness

    def fitness(self, gene, svm_acc, svm_weight, feature_weight, C):
        """ calculate the fitness of individuals """

        _fitness = svm_weight * svm_acc + feature_weight / float(sum(C * gene[0:self.feature_num]))
        return _fitness

    def select_population(self, population, fitness, keep_prob):
        """ select superior group with roulette """  # 基于轮盘赌选择（roulette selection）的种群选择过程

        prob_fitness = fitness / np.sum(fitness)
        cumsum_fitness = np.cumsum(prob_fitness)  #累积概率
        rands = [random.random() for _ in range(len(fitness))]

        new_population_index = []
        for i, rand in enumerate(rands):
            for j, prob in enumerate(cumsum_fitness):
                if rand <= prob:
                    new_population_index.append(j)
                    break
        new_population_index = np.asarray(new_population_index)

        keep_population_num = math.ceil(len(new_population_index) * keep_prob)

        # the bigger the probability, the easier the individual to be get
        selected_population_index = np.random.choice(a=new_population_index, size=keep_population_num, replace=False)

        return population[selected_population_index, :]

    def gene_cross(self, population, cross_prob):
        """gene cross, if cross gene, it will choice two parents randomly, and generate two new generations"""

        gene_num = len(population[0])
        new_generation = np.zeros((1, gene_num))

        new_num = math.ceil(cross_prob * self.length / 2)
        i = 0

        while 1 > 0:
            if i < new_num:
                parents_index = np.random.choice(len(population), 2, replace=False)
                parents = population[parents_index, :]
                cross_point_1 = random.randint(0, gene_num - 2)
                cross_point_2 = random.randint(cross_point_1 + 1, gene_num)  # 随机选择两个交叉点

                tmp = parents[1, cross_point_1:cross_point_2].copy()
                parents[1, cross_point_1:cross_point_2] = parents[0, cross_point_1:cross_point_2]
                parents[0, cross_point_1:cross_point_2] = tmp

                new_generation = np.concatenate([new_generation, parents], axis=0)
                i += 1
            else:
                break
        if new_generation.any() == 0:
            warnings.warn("No cross in population!", UserWarning)
            return population
        else:
            new_generation = np.delete(new_generation, 0, axis=0)
            new_generation = np.concatenate([new_generation, population], axis=0)
            return new_generation

    def gene_mutate(self, population, mutate_prob):
        """mutate gene with specific probability, it will randomly choose split point"""

        population_size = len(population)
        gene_num = len(population[0])
        for i in range(population_size):
            rand = random.random()
            if rand <= mutate_prob:
                mutate_point = random.randint(0, gene_num - 1)
                if population[i, mutate_point] == 0:
                    population[i, mutate_point] = 1
                else:
                    population[i, mutate_point] = 0

        return population

    def generate_population(self, size, feature_num):
        """generate population with populatition size and feature size"""
        population = np.zeros((size, feature_num), dtype=int)
        j = 0
        for i in range(len(population)):
            if i % 10 == 0:
                j += 1
            select_index = np.random.choice(feature_num, j, replace=False)
            population[i][select_index] = 1

        # 从种群矩阵中筛选出至少包含一个非零特征的个体
        population = population[np.where(population[:, 0:feature_num].any(axis=1))[0], :]

        return population

    def features_and_params_generator(self, X, y, population, feature_num):
        """针对每个个体，返回他所对应的数据集 （原始特征的子集， 样本全选 ， 行全选 列挑一部分）"""

        feature_genes = population[:, 0:feature_num]

        for i in range(len(population)):
            _features = X[:, np.where(feature_genes[i] == 1)[0]]

            yield _features, y

class GAForDenseNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self, 
        config: DenseNetTrainerConfig
    ):
        self.extractor = GAFeatureExtractor(config=config)

    def extract(
        self, 
        inputs: Union[np.ndarray, pd.DataFrame, str],
        trainer: Callable = None,
        **kwargs: Any
    ) -> GADenseNetFeatureExtractorOutput:
        output = self.extractor(
            inputs=inputs,
            trainer=trainer,
        )
        return output