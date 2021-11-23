import networks
import torch
import torch.nn as nn

class DRIT(nn.Module):
  def __init__(self, opts):
    super(DRIT, self).__init__()

    # parameters
    lr = 0.0001
    lr_dcontent = lr / 2.5
    self.nz = 8
    self.concat = opts.concat
    self.no_ms = opts.no_ms
    self.lambda_paired_L1 = opts.lambda_paired_L1
    self.lambda_paired_L1_random = opts.lambda_paired_L1_random
    self.lambda_L1_random_autoencoder = opts.lambda_L1_random_autoencoder
    self.lambda_paired_zc = opts.lambda_paired_zc
    self.lambda_paired_embedding = opts.lambda_paired_embedding
    self.lambda_D_content = opts.lambda_D_content
    self.dis_paired = opts.dis_paired
    self.dis_paired_neg_examples = opts.dis_paired_neg_examples
    self.contrastive_paired_zc = opts.contrastive_paired_zc
    self.contrastive_margin = opts.contrastive_margin

    # discriminators
    dis_dim_a = opts.input_dim_a
    dis_dim_b = opts.input_dim_b

        
    if opts.dis_scale > 1:
      self.disA = networks.MultiScaleDis(dis_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(dis_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.disA = networks.Dis(dis_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(dis_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)

    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)

    # generator
    if self.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    if self.contrastive_paired_zc:
        self.criterionTriplet = nn.TripletMarginLoss(margin=self.contrastive_margin, p=opts.triplet_distance_norm)

  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.gen.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def test_forward(self, image, a2b=True):
    self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    if a2b:
        self.z_content = self.enc_c.forward_a(image)
        output = self.gen.forward_b(self.z_content, self.z_random)
    else:
        self.z_content = self.enc_c.forward_b(image)
        output = self.gen.forward_a(self.z_content, self.z_random)
    return output

  def test_forward_transfer(self, image_a, image_b, a2b=True):
    self.z_content_a, self.z_content_b = self.enc_c.forward(image_a, image_b)
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a.forward(image_a, image_b)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a.forward(image_a, image_b)
    if a2b:
      output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
    else:
      output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
    return output

  def forward(self):
    # input images
    half_size = 1
    real_A = self.input_A
    real_B = self.input_B
    self.real_A_encoded = real_A[0:half_size]
    self.real_A_random = real_A[half_size:]
    self.real_B_encoded = real_B[0:half_size]
    self.real_B_random = real_B[half_size:]

    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)
    if self.contrastive_paired_zc:
        # get extra content embeddings for contrastive training
        self.z_content_a_random, self.z_content_b_random = self.enc_c.forward(self.real_A_random, self.real_B_random)
    

    # get random z_a
    self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

    # random-attr autoencoder (fake_AA_random and fake_BB_random) and cross-coder (fake_A_random and fake_B_random)
    input_content_forA = torch.cat((self.z_content_a, self.z_content_b),0)
    input_content_forB = torch.cat((self.z_content_b, self.z_content_a),0)
    input_attr_forA = torch.cat((self.z_random, self.z_random),0)
    input_attr_forB = torch.cat((self.z_random, self.z_random),0)
    output_A = self.gen.forward_a(input_content_forA, input_attr_forA)
    output_B = self.gen.forward_b(input_content_forB, input_attr_forB)
    self.fake_AA_random, self.fake_A_random = torch.split(output_A, self.z_content_a.size(0),dim=0)
    self.fake_BB_random, self.fake_B_random = torch.split(output_B, self.z_content_b.size(0),dim=0)

    # for display
    self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), \
                                    self.fake_AA_random[0:1].detach().cpu(), \
                                    self.fake_A_random[0:1].detach().cpu(), \
                                    self.real_B_encoded[0:1].detach().cpu(), 
                                    self.fake_BB_random[0:1].detach().cpu(), \
                                    self.fake_B_random[0:1].detach().cpu() ), dim=0)


  def forward_content(self):
    half_size = 1
    self.real_A_encoded = self.input_A[0:half_size]
    self.real_B_encoded = self.input_B[0:half_size]
    # get encoded z_c
    self.z_content_a, self.z_content_b = self.enc_c.forward(self.real_A_encoded, self.real_B_encoded)


  def update_D(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_AA_random)
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_BB_random)
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake = netD.forward(fake.detach())
    pred_real = netD.forward(real)
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G, Ec
    self.enc_c_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.enc_c_opt.step()
    self.gen_opt.step()


  def backward_EG(self):

    # Ladv for generator
    loss_G_GAN_A = self.backward_G_GAN(self.fake_AA_random, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_BB_random, self.disB)


    # random-attribute autoencoder reconstruction loss
    loss_G_L1_AA_random = self.lambda_L1_random_autoencoder*self.criterionL1(self.fake_AA_random, self.real_A_encoded)
    loss_G_L1_BB_random = self.lambda_L1_random_autoencoder*self.criterionL1(self.fake_BB_random, self.real_B_encoded)


    # contrastive paired content embedding loss
    loss_zc_paired = 0
    if self.contrastive_paired_zc:
        loss_zc_paired = self.lambda_paired_zc * ( \
                            self.criterionTriplet( self.z_content_a, self.z_content_b, self.z_content_a_random ) \
                            + self.criterionTriplet( self.z_content_a, self.z_content_b, self.z_content_b_random ) \
                            + self.criterionTriplet( self.z_content_a_random, self.z_content_b_random, self.z_content_a ) \
                            + self.criterionTriplet( self.z_content_a_random, self.z_content_b_random, self.z_content_b ) )


    loss_G = loss_G_GAN_A + loss_G_GAN_B + \
             loss_G_L1_AA_random + loss_G_L1_BB_random + \
             loss_zc_paired
            

    loss_G.backward(retain_graph=True)

    # validation losses (not used for backprop)
    # paired image translation loss
    loss_val_L1_paired_A = self.criterionL1(self.real_A_encoded, self.fake_A_random)
    loss_val_L1_paired_B = self.criterionL1(self.real_B_encoded, self.fake_B_random)

    self.gan_loss_a = loss_G_GAN_A.item()
    self.gan_loss_b = loss_G_GAN_B.item()

    self.l1_recon_AA_random_loss = loss_G_L1_AA_random
    self.l1_recon_BB_random_loss = loss_G_L1_BB_random

    self.l1_paired_A_val_loss = loss_val_L1_paired_A
    self.l1_paried_B_val_loss = loss_val_L1_paired_B

    

    if self.lambda_paired_zc > 0:
        self.zc_paired_loss = loss_zc_paired.item()

    self.G_loss = loss_G.item()




  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD.forward(fake)
    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G



  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.enc_c_sch.step()
    self.gen_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir)
    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disB.load_state_dict(checkpoint['disB'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.gen.load_state_dict(checkpoint['gen'])
    # optimizer
    if train:
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
         'disA': self.disA.state_dict(),
         'disB': self.disB.state_dict(),
         'enc_c': self.enc_c.state_dict(),
         'gen': self.gen.state_dict(),
         'disA_opt': self.disA_opt.state_dict(),
         'disB_opt': self.disB_opt.state_dict(),
         'enc_c_opt': self.enc_c_opt.state_dict(),
         'gen_opt': self.gen_opt.state_dict(),
         'ep': ep,
         'total_it': total_it
          }
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach()
    images_b = self.normalize_image(self.real_B_encoded).detach()
    images_a2 = self.normalize_image(self.fake_AA_random).detach()
    images_b2 = self.normalize_image(self.fake_BB_random).detach()
    images_a3 = self.normalize_image(self.fake_A_random).detach()
    images_b3 = self.normalize_image(self.fake_B_random).detach()
    row1 = torch.cat((images_a[0:1, ::], images_a2[0:1, ::], images_a3[0:1, ::]),3)
    row2 = torch.cat((images_b[0:1, ::], images_b2[0:1, ::], images_b3[0:1, ::]),3)
    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
