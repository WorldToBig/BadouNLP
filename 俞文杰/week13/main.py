import logging
import os

import numpy as np
import torch
from peft import get_peft_model, LoraConfig

from config import Config
from evaluate import Evaluator
from loader import load_data
from model import TorchModel, choose_optimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def peft_wrapper(model):
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )
    return get_peft_model(model, peft_config)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    model = peft_wrapper(model)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)

    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)
