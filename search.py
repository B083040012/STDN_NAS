from model import STDN_NAS
from dataloader import STDN_dataloader
from ASAGA import ASAGA_Searcher
import yaml, torch, logging, time
import torch.nn as nn
import numpy as np

if __name__=='__main__':
    
    # load yaml
    with open("parameters.yml", "r") as stream:
        config=yaml.load(stream, Loader=yaml.FullLoader)

    # load log file
    file_handlers=[
            logging.FileHandler(config["file"]["log_dir"]+"STDN_NAS.log"),
            logging.StreamHandler()
    ]

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt='%m/%d %I:%M:%S %p',
        handlers=file_handlers
    )

    """
    Load the pretrained supernet (in checkpoint.pth)
    """
    logging.info("[Architecture Searching Phase...]")
    lstm_seq_len=config["model"]["lstm_seq_len"]
    devices=config["model"]["device"]

    # loading model and criterion
    model=STDN_NAS(lstm_seq_len).to(devices)
    ckpt=torch.load("log\\checkpoint.pth", map_location=devices)
    model.load_state_dict(ckpt, strict=True)
    logging.info("Finishing import the pretrained supernet")
    criterion=nn.MSELoss()

    # loading val dataset
    null, val_loader, null=STDN_dataloader("train", config)

    """
    Searching for the Best Architecture
    by ASAGA
    """

    logger=logging.getLogger('Searcher')

    start=time.time()
    searcher=ASAGA_Searcher(config, logger, model, val_loader)
    searched_architecture, loss=searcher.search()
    end=time.time()
    logging.info("Search Complete")
    logging.info("[Searched Architecture Saved] Total Searched Time: %.5f sec, Architecture loss:%.5f" %((end-start), loss))
    np.save("searched_architecture.npz", searched_architecture)