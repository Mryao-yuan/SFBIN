import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import cv2



class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # analysis the pred result
        self.COLOR_MAP = {'0': (0, 0, 0), # black is TN
                          '1': (255, 255, 0), # yellow is FP 误检
                          '2': (255, 0, 0), # red is FN 漏检
                          '3':(0, 255, 0)} # green is TP

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)
        if os.path.exists(os.path.join(self.vis_dir,'pred')) is False:
            os.mkdir(os.path.join(self.vis_dir,'pred'))
        if os.path.exists(os.path.join(self.vis_dir,'analyse')) is False:
            os.mkdir(os.path.join(self.vis_dir,'analyse'))
        if os.path.exists(os.path.join(self.vis_dir,'gt')) is False:
            os.mkdir(os.path.join(self.vis_dir,'gt'))
        if os.path.exists(os.path.join(self.vis_dir,'compare')) is False:
            os.mkdir(os.path.join(self.vis_dir,'compare'))
        if os.path.exists(os.path.join(self.vis_dir,'t1')) is False:
            os.mkdir(os.path.join(self.vis_dir,'t1'))
        if os.path.exists(os.path.join(self.vis_dir,'t2')) is False:
            os.mkdir(os.path.join(self.vis_dir,'t2'))
        
    def gray2rgb(self, grayImage: np.ndarray, color_type: list):
        rgbImg = np.zeros((grayImage.shape[0], grayImage.shape[1], 3), dtype=np.uint8)
        grayImage=grayImage[:,:,1] # 输入灰度图像三通道,压缩为单通道
        for type_ in color_type:
            row,col= np.where(grayImage == type_)
            if (len(row) == 0):
                continue
            color = self.COLOR_MAP[str(type_)]
            rgbImg[row, col] = color
        return rgbImg
    def res_copare_pixel(self, pred, gt):
        assert np.max(pred) < self.n_class, 'pred must be in range [0, %d]' % self.n_class
        assert np.max(gt) < self.n_class, 'gt must be in range [0, %d]' % self.n_class
        visual_gray =self.n_class * gt.astype(int) + pred.astype(int)
        visual_rgb = self.gray2rgb(visual_gray,color_type=list(range(self.n_class**2)))
        return visual_rgb
    
    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1 and self.batch_id != 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)
        #  save the image
        if np.mod(self.batch_id, 100) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
            vis_pred = utils.make_numpy_grid(self._visualize_pred())
            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            vis_gt = np.clip(vis_gt, a_min=0.0, a_max=1.0)
            # # todo save gt
            # file_name = os.path.join(
            #     self.vis_dir,'gt' ,'gt_' + str(self.batch_id)+'.jpg')
            # plt.imsave(file_name, vis_gt)
            # # todo save the t1 t2 pred gt img
            # file_name = os.path.join(
            #     self.vis_dir, 'compare',str(self.batch_id)+'.jpg')
            # plt.imsave(file_name, vis)
            # # TODO save the t1 t2 img
            # file_name = os.path.join(self.vis_dir,'t1',str(self.batch_id)+'.jpg')
            # plt.imsave(file_name,vis_input)
            # file_name = os.path.join(self.vis_dir,'t2',str(self.batch_id)+'.jpg')
            # plt.imsave(file_name,vis_input2)
            # # todo save the pred img
            vis_pred = np.clip(vis_pred, a_min=0.0, a_max=1.0)
            pred_file_name = os.path.join(
                self.vis_dir, 'pred','eval_' + str(self.batch_id)+'_pred.jpg')
            plt.imsave(pred_file_name, vis_pred)
            # # todo save analyse pred img
            vis_analyse = self.res_copare_pixel(pred=vis_pred, gt=vis_gt)
            plt.imsave(os.path.join(self.vis_dir,'analyse', 'eval_' + str(self.batch_id)+'_analyse.jpg'), vis_analyse)

    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        # with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
        #           mode='a') as file:
        #     pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)[-1]

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
        

