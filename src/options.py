import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--paired_dataset', action='store_true', help='load paired data (pix2pix format)')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--lambda_paired_L1', type=float, default=0.0, help='paired image reconstruction loss')
    self.parser.add_argument('--lambda_paired_L1_random', type=float, default=0.0, help='paired image reconstruction loss for random-attribute images')
    self.parser.add_argument('--lambda_L1_random_autoencoder', type=float, default=0.0, help='random-attribute autoencoder reconstruction loss')
    self.parser.add_argument('--lambda_perceptual_random_autoencoder', type=float, default=0.0, help='random-attribute autoencoder perceptual loss')
    self.parser.add_argument('--lambda_paired_zc', type=float, default=0.0, help='paired image content encoder loss')
    self.parser.add_argument('--lambda_paired_embedding', type=float, default=0.0, help='generator paired embedding loss')
    self.parser.add_argument('--lambda_D_content', type=float, default=1.0, help='content discriminator adversarial loss weight')
    self.parser.add_argument('--dis_paired', action='store_true', help='use pix2pix-style paired discriminators')
    self.parser.add_argument('--dis_paired_neg_examples', action='store_true', help='train paired discr with examples of real, mismatched pairs')
    self.parser.add_argument('--contrastive_paired_zc', action='store_true', help='for paired images, use contrastive loss to train content encoder')
    self.parser.add_argument('--contrastive_margin', type=float, default=1.0, help='triplet loss margin')
    self.parser.add_argument('--triplet_distance_norm', type=int, default=2, help='triplet loss distance norm')
    self.parser.add_argument('--lambda_paired_zc_L1', type=float, default=0.0, help='paired L1 loss on content encoder')
    self.parser.add_argument('--lambda_adversarial', type=float, default=1.0, help='adversarial loss weight')

  def parse(self, options_string=None):
    self.opt = self.parser.parse_args(options_string)
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--paired_dataset', action='store_true', help='load paired data (pix2pix format)')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='../outputs', help='path for saving result images and models')

    # model related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

  def parse(self, options_string=None):
    self.opt = self.parser.parse_args(options_string)
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt
