import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__() # 继承父类
        self.main = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()  # 最后必须用tanh，把数据分布到（-1，1）之间
        )
    def forward(self, x):  # x表示长度为100的噪声输入
        img = self.main(x)
        img = img.view(-1, 28, 28, 1) # 方便等会绘图
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(), # x小于零是是一个很小的值不是0，x大于0是还是x
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid() # 保证输出范围为（0，1）的概率
        )
    def forward(self, x): # x表示28*28的mnist图片
        img = x.view(-1, 28*28)
        img = self.main(img)
        return img

def gen_img_plot(model, test_input):
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1) # 四行四列的第一个
        # imshow函数绘图的输入是（0，1）的float，或者（1，256）的int
        # 但prediction是tanh出来的范围是[-1，1]没法绘图，需要转成0~1(即加1除2)。
        plt.imshow( (prediction[i]+1)/2 )
        plt.axis('off')
    plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('training on ', device)
# 模型
gen = Generator().to(device)
dis = Discriminator().to(device)
# 优化器
g_opt = torch.optim.Adam(gen.parameters(), lr=0.0001)
d_opt = torch.optim.Adam(dis.parameters(), lr=0.0001)
# 损失
loss = torch.nn.BCELoss()

test_input = torch.randn(16, 100, device=device)

transform = transforms.Compose([
    transforms.ToTensor(),     # 归一化为0~1
    transforms.Normalize(0.5,0.5) # 归一化为-1~1
])
train_ds = torchvision.datasets.MNIST('datasets',  # 下载到那个目录下
                                      train=True,
                                      transform=transform,
                                      download=True)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
imgs, _ = next(iter(dataloader))
#imgs.shape    # torch.Size([64, 1, 28, 28])

D_loss = []
G_loss = []
epochs = 40
for epoch in range(epochs):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)  # 一个epoch的大小
    for step, (img, _) in enumerate(dataloader):
        img = img.to(device)  # 一个批次的图片
        size = img.size(0)  # 和和图片对应的原始噪音
        random_noise = torch.randn(size, 100, device=device)
        gen_img = gen(random_noise)  # 生成的图像

        d_opt.zero_grad()
        real_output = dis(img)  # 判别器输入真实图片，对真实图片的预测结果，希望是1
        # 判别器在真实图像上的损失
        d_real_loss = loss(real_output, torch.ones_like(real_output))  # size一样全一的tensor
        d_real_loss.backward()

        g_opt.zero_grad()
        # 记得切断生成器的梯度
        fake_output = dis(gen_img.detach())  # 判别器输入生成图片，对生成图片的预测结果，希望是0
        # 判别器在生成图像上的损失
        d_fake_loss = loss(fake_output, torch.zeros_like(fake_output))  # size一样全一的tensor
        d_fake_loss.backward()

        d_loss = d_real_loss + d_fake_loss
        d_opt.step()

        # 生成器的损失
        g_opt.zero_grad()
        fake_output = dis(gen_img)  # 希望被判定为1
        g_loss = loss(fake_output, torch.ones_like(fake_output))

        g_loss.backward()
        g_opt.step()

        # 每个epoch内的loss累加，循环外再除epoch大小，得到平均loss
        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss
    # 一个epoch训练完成
    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch: ', epoch)
        print('D_loss: ', D_loss)
        print('G_loss: ', G_loss)
        gen_img_plot(gen, test_input)

