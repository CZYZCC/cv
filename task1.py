import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import copy  
import glob
from torchvision.utils import save_image
from datetime import datetime
import logging


# 设置日志
def setup_logger():
    # 创建logs文件夹（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# 数据集类
class ImageTranslationDataset(Dataset):
    def __init__(self, root_dir="dataset", transform=None):
        self.transform = transform
        self.files = []
        
        # 遍历Path_1, Path_2, Path_3目录
        for path_num in range(1, 4):
            path_dir = os.path.join(root_dir, f"Path_{path_num}")
            real_dir = os.path.join(path_dir, "camera_images_real_front")
            semantic_dir = os.path.join(path_dir, "camera_images_semantic_front")
            
            # 获取所有图片文件
            real_images = glob.glob(os.path.join(real_dir, "*"))
            for real_path in real_images:
                # 获取对应的语义图像路径
                filename = os.path.basename(real_path)
                semantic_path = os.path.join(semantic_dir, filename)
                
                if os.path.exists(semantic_path):
                    self.files.append((real_path, semantic_path))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        real_path, semantic_path = self.files[idx]
        
        # 读取图像
        real_img = Image.open(real_path).convert('RGB')
        semantic_img = Image.open(semantic_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            real_img = self.transform(real_img)
            semantic_img = self.transform(semantic_img)
        
        return real_img, semantic_img

# U-Net的下采样块
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# U-Net的上采样块
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# Generator
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# 训练函数
def train(generator, discriminator, train_loader, val_loader, num_epochs, device, logger):
    # 创建保存模型和图像的目录
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_images", exist_ok=True)

    # 损失函数
    criterion_GAN = nn.MSELoss()
    criterion_pixelwise = nn.L1Loss()
    lambda_pixel = 100

    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 学习率调度器
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', patience=5, factor=0.5)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', patience=5, factor=0.5)
    
    # 早停设置
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_generator = None
    best_discriminator = None
    
    # 动态获取判别器输出尺寸
    sample_real, sample_semantic = next(iter(train_loader))
    sample_real, sample_semantic = sample_real[:1].to(device), sample_semantic[:1].to(device)
    with torch.no_grad():
        sample_gen = generator(sample_semantic)
        sample_output = discriminator(sample_semantic, sample_gen)
        output_shape = sample_output.shape
    logger.info(f"Discriminator output shape: {output_shape}")

    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        generator.train()
        discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for i, (real_imgs, semantic_imgs) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            
            # 转移数据到设备
            real_imgs = real_imgs.to(device)
            semantic_imgs = semantic_imgs.to(device)

            # 真实标签和假标签
            valid = torch.ones((batch_size, *output_shape[1:]), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, *output_shape[1:]), requires_grad=False).to(device)

            # 训练生成器
            optimizer_G.zero_grad()
            
            # 生成图像
            gen_imgs = generator(semantic_imgs)
            
            # 生成器损失
            pred_fake = discriminator(semantic_imgs, gen_imgs)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(gen_imgs, real_imgs)
            
            loss_G = loss_GAN + lambda_pixel * loss_pixel
            
            loss_G.backward()
            optimizer_G.step()

            # 训练判别器
            optimizer_D.zero_grad()
            
            # 真实图像损失
            pred_real = discriminator(semantic_imgs, real_imgs)
            loss_real = criterion_GAN(pred_real, valid)
            
            # 生成图像损失
            pred_fake = discriminator(semantic_imgs, gen_imgs.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            
            # 总判别器损失
            loss_D = 0.5 * (loss_real + loss_fake)
            
            loss_D.backward()
            optimizer_D.step()

            # 记录损失
            total_g_loss += loss_G.item()
            total_d_loss += loss_D.item()

            if i % 100 == 0:
                logger.info(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                    f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]"
                )

        # 计算平均训练损失
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)

        # 验证阶段
        generator.eval()
        discriminator.eval()
        val_g_loss = 0
        val_d_loss = 0
        
        with torch.no_grad():
            for real_imgs, semantic_imgs in val_loader:
                batch_size = real_imgs.size(0)
                real_imgs = real_imgs.to(device)
                semantic_imgs = semantic_imgs.to(device)
                
                # 真实和假标签
                valid = torch.ones((batch_size, *output_shape[1:]), requires_grad=False).to(device)
                fake = torch.zeros((batch_size, *output_shape[1:]), requires_grad=False).to(device)
                
                # 生成图像
                gen_imgs = generator(semantic_imgs)
                
                # 生成器损失
                pred_fake = discriminator(semantic_imgs, gen_imgs)
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_pixel = criterion_pixelwise(gen_imgs, real_imgs)
                loss_G = loss_GAN + lambda_pixel * loss_pixel
                
                # 判别器损失
                pred_real = discriminator(semantic_imgs, real_imgs)
                loss_real = criterion_GAN(pred_real, valid)
                pred_fake = discriminator(semantic_imgs, gen_imgs)
                loss_fake = criterion_GAN(pred_fake, fake)
                loss_D = 0.5 * (loss_real + loss_fake)
                
                val_g_loss += loss_G.item()
                val_d_loss += loss_D.item()
        
        # 计算平均验证损失
        avg_val_g_loss = val_g_loss / len(val_loader)
        avg_val_d_loss = val_d_loss / len(val_loader)
        
        # 更新学习率
        scheduler_G.step(avg_val_g_loss)
        scheduler_D.step(avg_val_d_loss)
        
        # 记录训练和验证损失
        logger.info(
            f"[Epoch {epoch}/{num_epochs}] "
            f"[Train G loss: {avg_g_loss:.4f}] [Train D loss: {avg_d_loss:.4f}] "
            f"[Val G loss: {avg_val_g_loss:.4f}] [Val D loss: {avg_val_d_loss:.4f}]"
        )
        
        # 检查是否是最佳模型（基于生成器验证损失）
        if avg_val_g_loss < best_val_loss:
            best_val_loss = avg_val_g_loss
            best_generator = copy.deepcopy(generator.state_dict())
            best_discriminator = copy.deepcopy(discriminator.state_dict())
            early_stopping_counter = 0
            
            # 保存最佳模型
            torch.save(best_generator, "saved_models/best_generator.pth")
            torch.save(best_discriminator, "saved_models/best_discriminator.pth")
            logger.info(f"Best model saved with validation G loss: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            logger.info(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
        # 早停检查
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # 保存定期检查点
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"saved_models/generator_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"saved_models/discriminator_{epoch}.pth")

        # 保存生成的样本
        if epoch % 5 == 0:
            with torch.no_grad():
                # 生成并保存一批图像
                fake_imgs = generator(semantic_imgs[:4])
                img_sample = torch.cat((semantic_imgs[:4], fake_imgs.data[:4], real_imgs[:4]), -2)
                save_image(img_sample, f"saved_images/epoch_{epoch}.png", normalize=True)
    
    # 训练结束，加载最佳模型
    generator.load_state_dict(best_generator)
    discriminator.load_state_dict(best_discriminator)
    return generator, discriminator

def test(generator, test_loader, device, logger):
    generator.eval()
    test_loss = 0
    criterion_pixelwise = nn.L1Loss()
    
    with torch.no_grad():
        for i, (real_imgs, semantic_imgs) in enumerate(test_loader):
            real_imgs = real_imgs.to(device)
            semantic_imgs = semantic_imgs.to(device)
            
            gen_imgs = generator(semantic_imgs)
            loss = criterion_pixelwise(gen_imgs, real_imgs)
            test_loss += loss.item()
            
            # 保存一些测试结果
            if i % 20 == 0:
                img_sample = torch.cat((semantic_imgs[:4], gen_imgs.data[:4], real_imgs[:4]), -2)
                save_image(img_sample, f"saved_images/test_sample_{i}.png", normalize=True)
    
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.4f}")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 设置参数
    batch_size = 4
    num_epochs = 200
    image_size = 256

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建数据集
    dataset = ImageTranslationDataset(root_dir="dataset", transform=transform)
    
    # 分割数据集为训练集、验证集和测试集 (8:1:1)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Dataset size: {dataset_size} pairs")
    logger.info(f"Training set: {train_size} pairs")
    logger.info(f"Validation set: {val_size} pairs")
    logger.info(f"Test set: {test_size} pairs")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 初始化模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 开始训练
    generator, discriminator = train(generator, discriminator, train_loader, val_loader, num_epochs, device, logger)
    
    # 在测试集上评估模型
    logger.info("Evaluating model on test set...")
    test(generator, test_loader, device, logger)


if __name__ == "__main__":
    # 设置日志记录器
    logger = setup_logger()
    
    # 运行主程序
    main()