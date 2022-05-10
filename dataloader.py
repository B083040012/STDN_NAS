import json, random
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class BSSDataConcat(data.Dataset):
    def __init__(self, nbhd, flow, label, tranform=None):
        self.transform= tranform
        self.__xs=nbhd
        self.__ys=flow
        self.__zs=label

    def __getitem__(self, index):
        return (self.__xs[index], self.__ys[index], self.__zs[index])
    
    def __len__(self):
        return len(self.__xs)

def STDN_dataloader(phase, config):
    """
    define the dataloader of STDN input
    1. training & testing data:
        a. nbhd_input
        b. flow_input
    2. training & testing label
    both load by .npz file

    data_size:
        slim dataset:
            vol:  (1000,7,2,7,7)
            flow: (1000,7,4,7,7)
            label:(1000,2)
            7 lstm_seq_len, 1000 input data, 2 or 4 channel, 7*7 image
        complete dataset:
            vol:  (287400,7,2,7,7)
            flow: (287400,7,4,7,7)
            label:(287400,2)
    """
    if phase=='train':
        loader=STDN_NAS_dataloader()
        train_nbhd_input, train_flow_input, train_label=loader.sample_stdn( datatype="train", \
                                                                            dataset_size=config["dataset"]["dataset_size"], \
                                                                            att_lstm_num=config["dataset"]["att_lstm_num"], \
                                                                            long_term_lstm_seq_len=config["dataset"]["long_term_lstm_seq_len"],
                                                                            short_term_lstm_seq_len=config["dataset"]["short_term_lstm_seq_len"], \
                                                                            nbhd_size=config["dataset"]["nbhd_size"],
                                                                            cnn_nbhd_size=config["dataset"]["cnn_nbhd_size"] )

        train_dataset=BSSDataConcat(train_nbhd_input, train_flow_input, train_label)
        return _get_STDN_dataloader(phase, train_dataset, config["training"]["batch_size"], config["training"]["num_workers"], config["training"]["train_portion"])

    elif phase=='test':
        loader=STDN_NAS_dataloader()
        test_nbhd_input, test_flow_input, test_label=loader.sample_stdn( datatype="test", \
                                                                         dataset_size="max", \
                                                                         att_lstm_num=config["dataset"]["att_lstm_num"], \
                                                                         long_term_lstm_seq_len=config["dataset"]["long_term_lstm_seq_len"],
                                                                         short_term_lstm_seq_len=config["dataset"]["short_term_lstm_seq_len"], \
                                                                         nbhd_size=config["dataset"]["nbhd_size"],
                                                                         cnn_nbhd_size=config["dataset"]["cnn_nbhd_size"] )

        test_dataset=BSSDataConcat(test_nbhd_input, test_flow_input, test_label)
        return _get_STDN_dataloader(phase, test_dataset, config["training"]["batch_size"], config["training"]["num_workers"], config["training"]["train_portion"])

def _get_STDN_dataloader(datatype, dataset, batch_size, num_workers, train_portion):
    """
    convert STDN dataset into dataloader type, 
    and split the validation data from training data
    """
    if datatype=='train':
        test_loader=None
        if train_portion != 1:
            train_len = len(dataset)
            indices = list(range(train_len))
            # shuffle index to enhance the randomness
            random.shuffle(indices)
            split = int(np.floor(train_portion * train_len))
            train_idx, val_idx = indices[:split], indices[split:]

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader=DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=train_sampler,
                pin_memory=True)
            val_loader=DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=val_sampler,
                pin_memory=True)
        else:
            train_loader=DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True)
            val_loader = None
        
    if datatype=='test':
        train_loader=None
        val_loader=None
        test_loader=DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader, test_loader

class STDN_NAS_dataloader:
    def __init__(self, config_path = "data\\original\\data_bike.json"):
        self.config = json.load(open(config_path, "r"))
        # how many timeslots per day (48 here)
        self.timeslot_daynum = int(86400 / self.config["timeslot_sec"])
        self.threshold = int(self.config["threshold"])
        # self.isVolumeLoaded = False
        # self.isFlowLoaded = False

    def load_train(self):
        volume_train = np.load(open(self.config["volume_train"], "rb"))["volume"] / self.config["volume_train_max"]
        flow_train = np.load(open(self.config["flow_train"], "rb"))["flow"] / self.config["flow_train_max"]
        return volume_train, flow_train

    def load_test(self):
        volume_test = np.load(open(self.config["volume_test"], "rb"))["volume"] / self.config["volume_train_max"]
        flow_test = np.load(open(self.config["flow_test"], "rb"))["flow"] / self.config["flow_train_max"]
        return volume_test, flow_test

    # def load_flow(self):
    #     self.flow_train = np.load(open(self.config["flow_train"], "rb"))["flow"] / self.config["flow_train_max"]
    #     self.flow_test = np.load(open(self.config["flow_test"], "rb"))["flow"] / self.config["flow_train_max"]
    #     self.isFlowLoaded = True

    # def load_volume(self):
    #     # shape (timeslot_num, x_num, y_num, type=2)
    #     self.volume_train = np.load(open(self.config["volume_train"], "rb"))["volume"] / self.config["volume_train_max"]
    #     self.volume_test = np.load(open(self.config["volume_test"], "rb"))["volume"] / self.config["volume_train_max"]
    #     self.isVolumeLoaded = True

    #this function nbhd for cnn, and features for lstm, based on attention model
    def sample_stdn(self, datatype, dataset_size, att_lstm_num = 3, long_term_lstm_seq_len = 3, short_term_lstm_seq_len = 7,\
    hist_feature_daynum = 7, last_feature_num = 48, nbhd_size = 2, cnn_nbhd_size = 3):

        if long_term_lstm_seq_len % 2 != 1:
            print("Att-lstm seq_len must be odd!")
            raise Exception

        if datatype == "train":
            data, flow_data=self.load_train()
        elif datatype == "test":
            data, flow_data=self.load_test()
        else:
            print("Please select **train** or **test**")
            raise Exception

        time_start = (hist_feature_daynum + att_lstm_num) * self.timeslot_daynum + long_term_lstm_seq_len
        if dataset_size=='max':
            time_end=data.shape[0]
        else:
            time_end=time_start+int(dataset_size/200)
        volume_type = data.shape[-1]

        """
        shape of the data:
            [time_range*region_num (input data num), short_term_lstm_seq_len, features_types, 2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1]
        shape of the label:
            [time_range*region_num, 2 (inflow & outflow)]
        """
        cnn_features = []
        flow_features = []
        for i in range((time_end-time_start)*(data.shape[1]*data.shape[2])):
            cnn_features.append([])
            flow_features.append([])
        labels = []

        for t in range(time_start, time_end):
            if t%100 == 0:
                print("Now sampling at {0} timeslots.".format(t))
            nbhd_count=0
            for x in range(data.shape[1]):
                for y in range(data.shape[2]):
                    
                    #sample common (short-term) lstm
                    short_term_lstm_samples = []
                    for seqn in range(short_term_lstm_seq_len):
                        # real_t from (t - short_term_lstm_seq_len) to (t-1)
                        real_t = t - (short_term_lstm_seq_len - seqn)

                        #cnn features, zero_padding
                        cnn_feature = np.zeros((volume_type, 2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1))
                        #actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                #boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                cnn_feature[0, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)] = data[real_t, cnn_nbhd_x, cnn_nbhd_y, 0]
                                cnn_feature[1, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)] = data[real_t, cnn_nbhd_x, cnn_nbhd_y ,1]
                        cnn_features[(t-time_start)*200+nbhd_count].append(cnn_feature)

                        #flow features, 4 type
                        flow_feature_curr_out = flow_data[0, real_t, x, y, :, :]
                        flow_feature_curr_in = flow_data[0, real_t, :, :, x, y]
                        flow_feature_last_out_to_curr = flow_data[1, real_t - 1, x, y, :, :]
                        #real_t - 1 is the time for in flow in longflow1
                        flow_feature_curr_in_from_last = flow_data[1, real_t - 1, :, :, x, y]

                        flow_feature = np.zeros(flow_feature_curr_in.shape+(4,))
                        
                        flow_feature[:, :, 0] = flow_feature_curr_out
                        flow_feature[:, :, 1] = flow_feature_curr_in
                        flow_feature[:, :, 2] = flow_feature_last_out_to_curr
                        flow_feature[:, :, 3] = flow_feature_curr_in_from_last
                        #calculate local flow, same shape cnn
                        local_flow_feature = np.zeros((4, 2*cnn_nbhd_size+1, 2*cnn_nbhd_size+1,))
                        #actual idx in data
                        for cnn_nbhd_x in range(x - cnn_nbhd_size, x + cnn_nbhd_size + 1):
                            for cnn_nbhd_y in range(y - cnn_nbhd_size, y + cnn_nbhd_size + 1):
                                #boundary check
                                if not (0 <= cnn_nbhd_x < data.shape[1] and 0 <= cnn_nbhd_y < data.shape[2]):
                                    continue
                                #get features
                                local_flow_feature[0, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, 0]
                                local_flow_feature[1, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, 1]
                                local_flow_feature[2, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, 2]
                                local_flow_feature[3, cnn_nbhd_x - (x - cnn_nbhd_size), cnn_nbhd_y - (y - cnn_nbhd_size)] = flow_feature[cnn_nbhd_x, cnn_nbhd_y, 3]
                        flow_features[(t-time_start)*200+nbhd_count].append(local_flow_feature)

                    #label
                    labels.append(data[t, x , y, :].flatten())
                    nbhd_count+=1

        for i in range((time_end-time_start)*(data.shape[1]*data.shape[2])):
            cnn_features[i] = np.array(cnn_features[i])
            flow_features[i] = np.array(flow_features[i])
        # short_term_lstm_features = np.array(short_term_lstm_features)
        labels = np.array(labels)

        return cnn_features, flow_features, labels