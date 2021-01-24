import sys
import torch
import torch.nn as nn
import numpy as np
from model.pose_generator_norm import Generator
from dataset.lisa_dataset_test import DanceDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import numpy as np
import math
import itertools
import time
import datetime
from scipy.io import wavfile
import pickle
import numpy as np
import librosa

from matplotlib import pyplot as plt
#import cv2
from shutil import copyfile
from dataset.output_helper import save_2_batch_images, save_batch_images, save_batch_images_combine
import argparse
from scipy.io.wavfile import write


def generate_old(args):
    file_path = args.model
    counter = args.count

    output_dir = args.output
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    audio_path = os.path.join(output_dir, "audio")
    try:
        os.makedirs(audio_path)
    except OSError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TensorNew = torch.FloatTensor()
    TensorNew.to(device)
    generator = Generator(1)
    generator.eval()
    generator.load_state_dict(torch.load(file_path, map_location='cpu'))
    generator.to(device)
    # data = DanceDataset(args.data)
    with open(args.wav, 'rb') as fo:  # 读取pkl文件数据
        data = pickle.load(fo)
    print(data.size())
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             # num_workers=8,
                                             pin_memory=False)

    criterion_pixelwise = torch.nn.L1Loss()
    count = 0
    total_loss = 0.0
    img_orig = np.ones((360, 640, 3), np.uint8) * 255
    for i, (x, target) in enumerate(dataloader):
        audio_out = x.view(-1)  # 80000
        scaled = np.int16(audio_out)

        audio = Variable(x.type(TensorNew).transpose(1, 0))  # 50,1,1600
        pose = Variable(target.type(TensorNew))  # 1,50,18,2
        pose = pose.view(1, 50, 36)

        # GAN loss
        fake = generator(audio)
        loss_pixel = criterion_pixelwise(fake, pose)
        total_loss += loss_pixel.item()

        fake = fake.contiguous().cpu().detach().numpy()  # 1,50,36
        fake = fake.reshape([50, 36])

        if (count <= counter):
            write(output_dir + "/audio/{}.wav".format(i), 16000, scaled)
            real_coors = pose.cpu().numpy()
            fake_coors = fake
            real_coors = real_coors.reshape([-1, 18, 2])
            fake_coors = fake_coors.reshape([-1, 18, 2])
            real_coors[:, :, 0] = (real_coors[:, :, 0] + 1) * 320
            real_coors[:, :, 1] = (real_coors[:, :, 1] + 1) * 180
            real_coors = real_coors.astype(int)

            fake_coors[:, :, 0] = (fake_coors[:, :, 0] + 1) * 320
            fake_coors[:, :, 1] = (fake_coors[:, :, 1] + 1) * 180
            fake_coors = fake_coors.astype(int)

            save_2_batch_images(real_coors, fake_coors, batch_num=count, save_dir_start=output_dir)
        count += 1

    final_loss = total_loss / count
    print("final_loss:", final_loss)


def generate_batch(args):
    file_path = args.model
    output_dir = args.output
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    audio_path = os.path.join(output_dir, "audio")
    try:
        os.makedirs(audio_path)
    except OSError:
        pass

    # data = DanceDataset(args.data)
    with open(args.wav, 'rb') as fo:  # 读取pkl文件数据
        data_ori = pickle.load(fo)
    data = (data_ori * 32768).int().float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TensorNew = torch.FloatTensor()
    TensorNew.to(device)
    generator = Generator(batch=data.shape[0])
    generator.eval()
    generator.load_state_dict(torch.load(file_path, map_location='cpu'))
    generator.to(device)

    data_ori_flat = data_ori.view(-1, 80000)
    audio_out = data.view(-1, 80000)  # 80000
    audio = Variable(data.transpose(1, 0))  # 50,1,1600

    # GAN loss
    fake = generator(audio)
    fake = fake.contiguous().cpu().detach().numpy()  # 1,50,36
    # fake = fake.reshape([50, 36])

    for i in range(fake.shape[0]):
        librosa.output.write_wav(os.path.join(output_dir, "audio/{}.wav").format(i), np.array(data_ori_flat[i, :]), 16000)
        fake_coors = fake[i,:,:]
        fake_coors = fake_coors.reshape([-1, 18, 2])
        fake_coors[:, :, 0] = (fake_coors[:, :, 0] + 1) * 320
        fake_coors[:, :, 1] = (fake_coors[:, :, 1] + 1) * 180
        fake_coors = fake_coors.astype(int)
        save_batch_images(fake_coors, batch_num=i, save_dir_start=output_dir)


    # dataloader = torch.utils.data.DataLoader(data,
    #                                          batch_size=1,
    #                                          shuffle=False,
    #                                          num_workers=1,
    #                                          # num_workers=8,
    #                                          pin_memory=False)
    #
    #
    # for i, x in enumerate(dataloader):
    #     audio_out = x.view(-1)  # 80000
    #     audio = Variable(x.transpose(1, 0))  # 50,1,1600
    #
    #     # GAN loss
    #     fake = generator(audio)
    #     fake = fake.contiguous().cpu().detach().numpy()  # 1,50,36
    #     fake = fake.reshape([50, 36])
    #
    #     librosa.output.write_wav(os.path.join(output_dir, "audio/{}.wav").format(i), np.array(audio_out), 16000)
    #     # write(os.path.join(output_dir, "audio/{}.wav").format(i), 16000, scaled)
    #     fake_coors = fake
    #     fake_coors = fake_coors.reshape([-1, 18, 2])
    #     fake_coors[:, :, 0] = (fake_coors[:, :, 0] + 1) * 320
    #     fake_coors[:, :, 1] = (fake_coors[:, :, 1] + 1) * 180
    #     fake_coors = fake_coors.astype(int)
    #     save_batch_images(fake_coors, batch_num=i, save_dir_start=output_dir)


def generate(args):
    file_path = args.model
    output_dir = args.output
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    audio_path = os.path.join(output_dir, "audio")
    try:
        os.makedirs(audio_path)
    except OSError:
        pass

    # data = DanceDataset(args.data)
    with open(args.wav, 'rb') as fo:  # 读取pkl文件数据
        data_ori = pickle.load(fo)
    data = (data_ori * 32768).int().float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TensorNew = torch.FloatTensor()
    TensorNew.to(device)
    generator = Generator(batch=data.shape[0])
    generator.eval()
    generator.load_state_dict(torch.load(file_path, map_location='cpu'))
    generator.to(device)

    data_ori_flat = data_ori.view(-1, 80000)
    audio_out = data.view(-1, 80000)  # 80000
    audio = Variable(data.transpose(1, 0))  # 50,1,1600

    # GAN loss
    fake = generator(audio)
    fake = fake.contiguous().cpu().detach().numpy()  # 1,50,36
    # fake = fake.reshape([50, 36])

    for i in range(fake.shape[0]):
        # librosa.output.write_wav(os.path.join(output_dir, "audio/{}.wav").format(i), np.array(data_ori_flat[i, :]), 16000)
        fake_coors = fake[i,:,:]
        fake_coors = fake_coors.reshape([-1, 18, 2])
        fake_coors[:, :, 0] = (fake_coors[:, :, 0] + 1) * 320
        fake_coors[:, :, 1] = (fake_coors[:, :, 1] + 1) * 180
        fake_coors = fake_coors.astype(int)
        # print(fake_coors)
        #
        # for row in range(fake_coors.shape[0]):
        #     for col in range(fake_coors.shape[1]):
        #         for c in range(fake_coors.shape[2]):
        #             pv = fake_coors[row, col, c]
        #             if c == 0:
        #                 fake_coors[row, col, c] = 320 - pv
        #             elif c == 1:
        #                 fake_coors[row, col, c] = 180 - pv

        save_batch_images_combine(fake_coors, batch_num=i, save_dir_start=output_dir)

    copyfile(src=args.wav[:-3]+'wav', dst=os.path.join(output_dir, "audio/0.wav"))




def audio_split(args):
    wav_path = args.wav
    pkl_path = wav_path.replace("wav", "pkl")
    args.wav = pkl_path
    if os.path.exists(pkl_path):
        return
    # input数据处理, 降采样到16k
    y, sr = librosa.load(wav_path)
    y_16k = librosa.resample(y, sr, 16000)
    print(y.shape, y_16k.shape)
    librosa.output.write_wav(wav_path, y_16k, 16000)

    # 剪切数据
    y_copy = torch.tensor(y_16k[:int(np.floor(len(y_16k) / (50 * 1600)) * (50 * 1600))])
    y_resize = y_copy.view(-1, 50, 1600)
    print(y_resize.size())

    output = open(pkl_path, 'wb')
    pickle.dump(y_resize, output)
    output.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wav",
        default="./input/export.wav",
        metavar="FILE",
        help="path to wav file",
        type=str
    )

    parser.add_argument(
        "--model",
        default="./pretrain_model/generator_0400.pth",
        metavar="FILE",
        help="path to pth file",
        type=str
    )

    parser.add_argument(
        "--data",
        default="./dataset/clean_revised_pose_pairs.json",
        metavar="FILE",
        help="path to pth file",
        type=str
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100
    )

    parser.add_argument(
        "--output",
        default="./output/",
        metavar="FILE",
        help="path to output",
        type=str
    )

    args = parser.parse_args()

    audio_split(args)
    generate(args)

