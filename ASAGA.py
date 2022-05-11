import numpy as np
import torch.nn as nn
from utils import AverageMeter
from train_supernet import eval_all
import torch, math


class ASAGA_Searcher():
    
    def __init__(self, config, logger, model, val_loader):
        self.logger=logger
        self.model=model
        self.val_loader=val_loader
        self.config=config
        self.generation_num=config["searching"]["generation_num"]
        self.population_num=config["searching"]["population_num"]
        self.annealing_ratio=config["searching"]["annealing_ratio"]
        self.initial_tmp=config["searching"]["initial_tmp"]
        self.crossover_rate=config["searching"]["crossover_rate"]
        self.num_choice=config["model"]["num_choice"]
        self.num_layers=config["model"]["num_layers"]
    def search(self):
        """
        Initialization
        1. initialize the population 
        2. calculate fitness for each architecture
        """
        parent_population=[]
        for p in range(self.population_num):
            architecture=list(np.random.randint(self.num_choice, size=self.num_layers))
            # no avaliable condition currently
            parent_population.append(architecture)
        parent_population=np.array(parent_population)
        parent_fitness=self.evaluate_architecture(parent_population, self.val_loader)
        # tmp_best_index=np.argmin(parent_fitness)
        # tmp_best_loss=parent_fitness[tmp_best_index]
        tmp_best_loss=min(parent_fitness)
        tmp_best_index=parent_fitness.index(tmp_best_loss)
        self.logger.info("[Population Initialize] tmp_best_loss: %.5f" %(tmp_best_loss))
        
        """
        Generation Start
        1. loop (n/2) times:
            (a) generate two offsprings from two randomly chosen parent by crossover
            (b) using SA to select the parent of offspring
            (c) overwrite the old architecture with selected architecture
        2. lower temperature T
        """
        self.curr_tmp=self.initial_tmp
        global_best_loss=tmp_best_loss
        global_best_architecture=parent_population[tmp_best_index]
        offspring_population=parent_population
        self.logger.info("--------------[Generation Start]--------------")
        for gen in range(self.generation_num):
            for loop in range(int(self.population_num/2)):
                index_list=[np.random.randint(low=0, high=self.num_layers),np.random.randint(low=0, high=self.num_layers)]
                parent_list=[parent_population[index] for index in index_list]
                parent_subfitness=[parent_fitness[index] for index in index_list]
                offspring_list=self.crossover(parent_list)
                offspring_subfitness=self.evaluate_architecture(offspring_list, self.val_loader)
                new_fitness=self.selection(parent_subfitness, offspring_subfitness, parent_population, offspring_list, index_list)
                for i in range(len(new_fitness)):
                    parent_fitness[index_list[i]]=new_fitness[i]
            # tmp_best_index=np.argmin(parent_fitness)
            # tmp_best_loss=parent_fitness[tmp_best_index]
            tmp_best_loss=min(parent_fitness)
            tmp_best_index=parent_fitness.index(tmp_best_loss)
            tmp_best_architecture=parent_population[tmp_best_index]
            if global_best_loss>tmp_best_loss:
                global_best_loss=tmp_best_loss
                global_best_architecture=tmp_best_architecture
                self.logger.info("%%%%%%%%%%%%%%%%%%%%%%%%")
                self.logger.info("[Best Loss] gen:%d, gloabl_best_loss:%.5f" %(gen, global_best_loss))
                self.logger.info("%%%%%%%%%%%%%%%%%%%%%%%%")
            if gen%10==0:
                self.logger.info("[Generation %3d] tmp:%.5f, tmp_best:%.5f, gloabl_best_loss:%.5f" %(gen, self.curr_tmp, tmp_best_loss, global_best_loss))
            self.curr_tmp=self.curr_tmp*self.annealing_ratio
        self.logger.info("--------------[Generation End]--------------")

        return global_best_architecture, global_best_loss
                
    def crossover(self, parent_list):
        """
        crossover on single point
        """
        cross_point=np.random.randint(low=0, high=self.num_layers)
        offspring_list=parent_list
        offspring_list[0][:cross_point]=parent_list[1][:cross_point]
        offspring_list[1][cross_point:]=parent_list[0][cross_point:]
        return offspring_list

    def selection(self, parent_subfitness, offspring_subfitness, parent_population, offspring_list, index_list):
        """
        select and overwrite
        """
        new_fitness=parent_subfitness
        for i in range(len(parent_subfitness)):
            prob=np.random.uniform(0,1)
            accept_prob=math.exp(-(offspring_subfitness[i]-parent_subfitness[i])/self.curr_tmp)
            if parent_subfitness[i]>offspring_subfitness[i]:
                parent_population[index_list[i]]=offspring_list[i]
                new_fitness[i]=offspring_subfitness[i]
            elif prob<accept_prob:
                parent_population[index_list[i]]=offspring_list[i]
                new_fitness[i]=offspring_subfitness[i]
        return new_fitness

    def evaluate_architecture(self, architecture_list, val_loader):
        """
        Evaluate architecture in population,
        return the loss value of each architecture
        """
        architecture_loss=[]
        criterion = nn.MSELoss()
        for index, architecture in enumerate(architecture_list):
            self.model.eval()
            total_val_loss=AverageMeter()
            
            with torch.no_grad():
                for (nbhd, flow, label) in val_loader:
                    # resize the tensor to [lstm_seq_num, batch_size, channel, size, size]
                    cnn_tensor_list=nbhd.permute(1,0,2,3,4)
                    flow_tensor_list=flow.permute(1,0,2,3,4)
                    target=label.to(self.config["model"]["device"]).float()

                    # load the architecture
                    output=self.model(cnn_tensor_list, flow_tensor_list, list(architecture))

                    # criterion-total part
                    output=output*self.config["dataset"]["vol_train_max"]
                    target=target*self.config["dataset"]["vol_train_max"]
                    threshold=self.config["dataset"]["threshold"]*self.config["dataset"]["vol_train_max"]
                    total_loss, validlen=eval_all(output, target, criterion, threshold)
                    if total_loss==-1:
                        continue
                    total_val_loss.update(total_loss.item(), validlen)
            architecture_loss.append(total_val_loss.avg)
        # architecture_loss=np.array(architecture_loss)
        return architecture_loss