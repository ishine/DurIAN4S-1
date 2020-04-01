import sys 
sys.path.append('model')
sys.path.append('extension')

import os 
import torch 
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataprocess
from file_xt import FileXT
from config_xt import ConfigXT
from torch_xt import set_device, save_checkpoint, LossLog
from tacotron import Tacotron

def main():
    config = ConfigXT()
    config_basename = FileXT(config.file).basename
    print("Configuration file: %s" % (config_basename))

    checkpoint_path = config.checkpoint_path
    if not config.test_run:
        checkpoint_path = FileXT(config.checkpoint_path, '').create_path()
        config.save(os.path.join(checkpoint_path, config_basename))
        writer = SummaryWriter(checkpoint_path)

    dataloader = dataprocess.load_train(config)
    model = Tacotron(config)
    model = set_device(model, config.device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=len(dataloader.train)*config.step_size, gamma=config.factor)

    losses = []
    loss_train = LossLog()
    loss_valid = LossLog()
    for epoch in range(config.stop_epoch):
        # Train Loop 
        model.train()
        for batch in tqdm(dataloader.train, leave=False, ascii=True):
            x, y_prev, y = set_device(batch, config.device)

            optimizer.zero_grad()
            y_gen, y_decoder_gen = model(x, y_prev)
            loss = criterion(y_gen, y) + criterion(y_decoder_gen, y)
            loss.backward()
            if config.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optimizer.step()
            scheduler.step()

            loss_train.add(loss.item(), y[0].size(0))
            if not config.test_run:
                writer.add_scalar('train/l1_loss', loss.item(), loss_train.iteration)

        # Validation Loop
        model.eval()
        for batch in tqdm(dataloader.valid, leave=False, ascii=True):
            x, y_prev, y = set_device(batch, config.device)

            y_gen, y_decoder_gen = model(x, y_prev)
            loss = criterion(y_gen, y) + criterion(y_decoder_gen, y)

            loss_valid.add(loss.item(), y[0].size(0))
            if not config.test_run:
                writer.add_scalar('valid/l1_loss', loss.item(), loss_valid.iteration)

        learn_rate = scheduler.get_lr()[0]
        print("[Epoch %d/%d] [loss train: %.5f] [loss valid: %.5f] [lr: %.5f]" %
             (epoch, config.stop_epoch, loss_train.avg(), loss_valid.avg(), learn_rate))

        losses.append([loss_train.avg(), loss_valid.avg()])
        loss_train.reset()
        loss_valid.reset()

        if not config.test_run: 
            loss_savename = os.path.join(checkpoint_path, 'loss.pt')
            torch.save(losses, loss_savename)

            if epoch%config.save_epoch == 0:
                savename = os.path.join(checkpoint_path, 'epoch' + str(epoch) + '.pt')
                save_checkpoint(savename, model, optimizer, learn_rate, loss_train.iteration)

if __name__ == "__main__":
    main()