import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


from network.encoder import SimpleViTEncoder

from config.config import load_config
from data.expert_dataset import ExpertDataset

config = load_config('encoder.yaml', 'encoder')

class EncoderDecoder(nn.Module):
    def __init__(self, image_size, channels) -> None:
        super(EncoderDecoder, self).__init__()

        self.encoder = SimpleViTEncoder(
            image_size=image_size,
            channels=channels,
            patch_size=config.patch_size,
            dim=config.dim,
            depth=config.n_layer,
            heads=config.n_head,
            mlp_dim=config.mlp_dim,
            dim_head=config.dim_head,
        )
        self.image_size = image_size
        self.channels = channels
        dim = config.dim

        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, image_size * image_size * channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(-1, self.channels, self.image_size, self.image_size)
        return x
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), str(path / "obs_encoder.pth"))
        torch.save(self.decoder.state_dict(), str(path / "obs_decoder.pth"))

    def load(self, path: Path):
        self.encoder.load_state_dict(torch.load(str(path / "obs_encoder.pth")))
        self.decoder.load_state_dict(torch.load(str(path / "obs_decoder.pth")))
        

class EncoderTrainer:
    def __init__(self, image_size, channels, lr, batch_size, device='cuda') -> None:
        self.data_loader = None
        self.device = device
        self.batch_size = batch_size
        self.model = EncoderDecoder(
            image_size=image_size,
            channels=channels
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def set_dataset(self, dataset: Dataset):
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def update(self):
        assert self.data_loader is not None, "Please set dataset first."
        # 迭代数据加载器以获取批次数据
        loss_sum = 0
        num = 0
        for batch in self.data_loader:
            # 在这里执行训练步骤，例如训练神经网络模型
            num += 1
            obs = batch['obs']
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            obs_preds = self.model(obs)
            loss = self.loss(obs_preds, obs)

            self.optimizer.zero_grad()
            loss.backward()
            loss_sum += loss.item()
            # print(loss.shape)
            self.optimizer.step()
        
        # self.scheduler.step()
        return loss_sum / num   
        
    
    def save(self, path: Path):
        self.model.save(path)

    def load(self, path: Path):
        self.model.load(path)

    def eval(self, path: Path, index):
        import cv2
        path.mkdir(parents=True, exist_ok=True)
        save_dir = str(path / f"eval_{index}.jpg")
        with torch.no_grad():
            for batch in self.data_loader:
                obs = torch.tensor(batch['obs'][0:1], dtype=torch.float32).to(self.device)
                obs_image = obs.cpu().squeeze() * 255
                obs_image = [obs_image[i*3:i*3+3] for i in range(3)][::-1]
                obs_image = torch.cat(obs_image, dim=2)

                encoder_image = self.model(obs).cpu().squeeze() * 255
                encoder_image = [encoder_image[i*3:i*3+3] for i in range(3)][::-1]
                encoder_image = torch.cat(encoder_image, dim=2)

                image = torch.cat((obs_image, encoder_image), dim=1).numpy().transpose(1, 2, 0).astype(np.uint8)
                # 保存图片
                cv2.imwrite(save_dir, image)
                # input()
                return

        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=80)
    parser.add_argument('--channels', type=int, default=9)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_name', type=list, default="")
    parser.add_argument('--cheakpoint_freq', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--logdir', type=str, default="log/conservative_encoder/")

    args = parser.parse_args()

    dataset = ExpertDataset()
    dataset.loade_dataset(args.data_name)
    dataset.start()
    
    trainer = EncoderTrainer(
        image_size=args.image_size,
        channels=args.channels,
        lr=args.lr,
        batch_size=args.batch_size
    )
    trainer.set_dataset(dataset)

    trainer.load(Path("log/conservative_encoder/cheakpoint"))

    print("\nStart training.\n")
    logdir = Path(args.logdir)
    save_dir = logdir / "encoder_trained"
    save_dir.mkdir(parents=True, exist_ok=True)
    cheakpoint_dir = logdir / "cheakpoint"
    cheakpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = logdir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir / 'tensorboard'))

    cheakpoint_freq = args.cheakpoint_freq
    epochs = args.epochs
    eval_freq = 50
    for index in range(epochs):
        print(f"\n=========Training, epoch {index}=========\n")
        loss = trainer.update()
        print(f"Loss: {loss}")
        writer.add_scalar('Loss/train', loss, index)
        if (index + 1) % cheakpoint_freq == 0:
            trainer.save(cheakpoint_dir)
        if (index + 1) % eval_freq == 0:
            trainer.eval(eval_dir, index + 1)
    trainer.save(save_dir)