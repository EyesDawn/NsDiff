from src.experiments.exp_point_basic import Exp_Point_Basic
from src.utils.tools import EarlyStopping, adjust_learning_rate, visual
from src.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch_timeseries.dataset import *
import torch_timeseries.dataset as dataset_module
from src.datasets import *
from torch_timeseries.scaler import *
from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader
from torch_timeseries.utils.parse_type import parse_type
try:
    import wandb
except:
    print("Warning: wandb is not installed, some functionality may not work.")
    wandb = None

class DatasetWrapper:
    def __init__(self, dataset, scaler):
        self.dataset = dataset
        self.scaler = scaler
        self.scale = True

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)

class LoaderWrapper:
    def __init__(self, loader, args):
        self.loader = loader
        self.args = args
        self.dataset = loader.dataset

    def __iter__(self):
        for batch in self.loader:
            batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc = batch
            
            label_len = self.args.label_len
            
            batch_y_start = batch_x[:, -label_len:, :]
            new_batch_y = torch.cat([batch_y_start, batch_y], dim=1)
            
            batch_y_mark_start = batch_x_date_enc[:, -label_len:, :]
            new_batch_y_mark = torch.cat([batch_y_mark_start, batch_y_date_enc], dim=1)
            
            yield batch_x, new_batch_y, batch_x_date_enc, new_batch_y_mark

    def __len__(self):
        return len(self.loader)


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Point_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.wandb_project = getattr(args, 'wandb_project', None)
        self.wandb_run = None
        if self.wandb_project and wandb is not None:
            # 初始化 wandb，但不在 __init__ 中创建 run，等到 train 方法中再创建
            pass

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if not hasattr(self, 'train_loader'):
            self._init_data_loader()
            
        if flag == 'test':
            return self.test_data, self.test_loader
        elif flag == 'val':
            return self.val_data, self.val_loader
        elif flag == 'pred':
            return self.test_data, self.test_loader
        else:
            return self.train_data, self.train_loader

    def _init_data_loader(self):
        dataset_type = self.args.data
        root_path = self.args.root_path
        
        # 创建一个包含所有必要模块的命名空间，确保能找到所有数据集类
        namespace = globals().copy()
        namespace.update(vars(dataset_module))
        
        self.dataset = parse_type(dataset_type, namespace)(
            root=root_path
        )
        
        scaler_type = getattr(self.args, 'scaler_type', 'StandardScaler')
        self.scaler = parse_type(scaler_type, namespace)()
        
        window = self.args.seq_len
        steps = self.args.pred_len
        horizon = getattr(self.args, 'horizon', 1)
        
        batch_size = self.args.batch_size
        num_worker = getattr(self.args, 'num_workers', 0)
        
        if dataset_type.startswith("ETT"):
            if dataset_type.startswith("ETTh"):
                Loader = ETTHLoader
            elif dataset_type.startswith("ETTm"):
                Loader = ETTMLoader
            else:
                Loader = SlidingWindowTS
                
            self.dataloader = Loader(
                self.dataset,
                self.scaler,
                window=window,
                horizon=horizon,
                steps=steps,
                shuffle_train=True,
                freq=self.dataset.freq,
                batch_size=batch_size,
                num_worker=num_worker,
                fast_test=False,
                fast_val=False,
            )
        else:
            self.dataloader = SlidingWindowTS(
                self.dataset,
                self.scaler,
                window=window,
                horizon=horizon,
                steps=steps,
                scale_in_train=True,
                shuffle_train=True,
                freq=self.dataset.freq,
                batch_size=batch_size,
                train_ratio=0.7,
                test_ratio=0.2,
                num_worker=num_worker,
                fast_test=False,
                fast_val=False,
            )
            
        self.train_loader = LoaderWrapper(self.dataloader.train_loader, self.args)
        self.val_loader = LoaderWrapper(self.dataloader.val_loader, self.args)
        self.test_loader = LoaderWrapper(self.dataloader.test_loader, self.args)
        
        self.train_data = DatasetWrapper(self.dataset, self.scaler)
        self.val_data = DatasetWrapper(self.dataset, self.scaler)
        self.test_data = DatasetWrapper(self.dataset, self.scaler)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'SolarEnergy' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 初始化 wandb
        if self.wandb_project and wandb is not None:
            wandb.init(
                project=self.wandb_project,
                name=setting,
                config={
                    'model': self.args.model,
                    'data': self.args.data,
                    'seq_len': self.args.seq_len,
                    'pred_len': self.args.pred_len,
                    'e_layers': self.args.e_layers,
                    'd_model': self.args.d_model,
                    'd_ff': self.args.d_ff,
                    'batch_size': self.args.batch_size,
                    'learning_rate': self.args.learning_rate,
                    'train_epochs': self.args.train_epochs,
                    'patience': self.args.patience,
                }
            )
            self.wandb_run = wandb.run

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler('cuda')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'SolarEnergy' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # 记录到 wandb
            if self.wandb_run is not None:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': vali_loss,
                    'test_loss': test_loss,
                    'epoch': epoch + 1
                })
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 完成 wandb run
        if self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        # 如果 wandb 还没有初始化（测试模式），则初始化它
        if self.wandb_project and wandb is not None and self.wandb_run is None:
            wandb.init(
                project=self.wandb_project,
                name=setting + '_test',
                config={
                    'model': self.args.model,
                    'data': self.args.data,
                    'seq_len': self.args.seq_len,
                    'pred_len': self.args.pred_len,
                    'e_layers': self.args.e_layers,
                    'd_model': self.args.d_model,
                    'd_ff': self.args.d_ff,
                }
            )
            self.wandb_run = wandb.run

        preds = []
        trues = []
        # folder_path = './results/visual_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'SolarEnergy' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    # 直接对整个批次进行 inverse_transform，不使用 squeeze(0)
                    # 这样可以保持批次维度一致
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         input = test_data.inverse_transform(input)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 使用 concatenate 而不是 array，这样可以处理不同批次大小的情况
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/runs/iTransformer/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("./results/runs/iTransformer/result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # 记录测试指标到 wandb
        if self.wandb_run is not None:
            wandb.log({
                'test_mae': mae,
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mape': mape,
                'test_mspe': mspe
            })
            wandb.run.summary.update({
                'test_mae': mae,
                'test_mse': mse,
                'test_rmse': rmse,
                'test_mape': mape,
                'test_mspe': mspe
            })

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # 如果是在测试模式下初始化的 wandb，则在这里完成
        if self.wandb_run is not None and test:
            wandb.finish()
            self.wandb_run = None

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return