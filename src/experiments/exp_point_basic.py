import os
import torch
from src.models import iTransformer


class Exp_Point_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # A batch scheduler may already mask one physical GPU through
            # CUDA_VISIBLE_DEVICES.  Keep that mask intact: inside such a
            # process the selected physical GPU is always logical cuda:0.
            # The previous unconditional assignment here replaced the mask
            # with argparse's default gpu=0 and sent every scheduled job to
            # physical GPU 0.
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices and not self.args.use_multi_gpu:
                device = torch.device('cuda:0')
                print(f'Use GPU: cuda:0 (CUDA_VISIBLE_DEVICES={visible_devices})')
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
