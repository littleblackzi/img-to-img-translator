from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')#每 400 次迭代在屏幕上显示一次训练结果（如图像）
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')#控制 Visdom 中每行显示的图像数量
        parser.add_argument('--display_id', type=int, default=None, help='window id of the web display. Default is random window id')#指定 Visdom 窗口 ID，默认随机
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')#Visdom 服务器地址，默认本地
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')#Visdom 的环境名，类似工作空间
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')#Visdom 默认端口 8097
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')#每 1000 次迭代将训练结果保存为 HTML 文件
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')#每 100 次迭代在控制台打印一次训练信息
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')#如果设置，就不保存 HTML 可视化结果
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')#每 5000 次迭代保存一次最新模型
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')#每 5 个 epoch 保存一次完整 checkpoint
        parser.add_argument('--evaluation_freq', type=int, default=5000, help='evaluation freq')#每 5000 次迭代进行一次评估
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')#如果设置，就按迭代次数保存模型，而不是按 epoch
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')#如果设置，就从最新 checkpoint 继续训练
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')#指定从第几个 epoch 开始计数
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')#当前阶段，训练、验证或测试
        parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from another checkpoint')#从其他实验的 checkpoint 继续训练

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs with the initial learning rate')#初始学习率持续 200 个 epoch
        parser.add_argument('--n_epochs_decay', type=int, default=200, help='number of epochs to linearly decay learning rate to zero')#之后再花 200 个 epoch 将学习率线性降到 0
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')#Adam 优化器的两个动量参数
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')#初始学习率 0.0002
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')#GAN 损失类型：L2 损失（LSGAN）、原始交叉熵（vanilla）或 WGAN-GP
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')#用于训练判别器的图像缓冲区大小（如 CycleGAN 的历史图像池）
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')#学习率衰减策略
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--cost_type', type=str, default='hard', help='hard weighting or easy weighting')#损失加权方式：hard 或 easy
        parser.add_argument('--eps', type=float, default=1.0, help='epsilon of OT')#最优传输（OT）中的正则化参数 ε
        parser.add_argument('--neg_term_weight', type=float, default=1.0, help='weight of negative term')#负样本项的权重

        self.isTrain = True
        return parser
