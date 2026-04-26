import argparse
import random

import numpy as np
import setproctitle
import torch

from src.experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from src.models.DLinear import Model as DLinearModel

try:
    import wandb
except Exception:
    print("Warning: wandb is not installed, some functionality may not work.")
    wandb = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class Exp_Long_Term_Forecast_DLinear(Exp_Long_Term_Forecast):
    def _build_model(self):
        model = DLinearModel(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


if __name__ == "__main__":
    setproctitle.setproctitle("DLinear")
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="DLinear")

    parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
    parser.add_argument("--model_id", type=str, required=True, default="test", help="model id")
    parser.add_argument("--model", type=str, required=True, default="DLinear", help="model name")

    parser.add_argument("--data", type=str, required=True, default="custom", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./data/electricity/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="electricity.csv", help="data csv file")
    parser.add_argument("--features", type=str, default="M", help="forecasting task")
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")

    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")

    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="conditioning feature dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="compatibility argument")
    parser.add_argument("--e_layers", type=int, default=2, help="compatibility argument")
    parser.add_argument("--d_layers", type=int, default=1, help="compatibility argument")
    parser.add_argument("--d_ff", type=int, default=2048, help="compatibility argument")
    parser.add_argument("--moving_avg", type=int, default=25, help="moving average window")
    parser.add_argument("--individual", type=str2bool, default=False, help="whether to use individual linear heads")
    parser.add_argument("--factor", type=int, default=1, help="compatibility argument")
    parser.add_argument("--distil", action="store_false", default=True, help="compatibility argument")
    parser.add_argument("--dropout", type=float, default=0.1, help="compatibility argument")
    parser.add_argument("--embed", type=str, default="timeF", help="compatibility argument")
    parser.add_argument("--activation", type=str, default="gelu", help="compatibility argument")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")
    parser.add_argument("--do_predict", action="store_true", help="whether to predict unseen future data")
    parser.add_argument("--use_norm", type=str2bool, default=True, help="use norm and denorm")

    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", "--epochs", dest="train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--lr_patience", type=int, default=2, help="patience for plateau lr scheduler")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="decay factor for plateau lr scheduler")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="minimum learning rate for plateau lr scheduler")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)

    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

    parser.add_argument("--exp_name", type=str, required=False, default="MTSF", help="experiment name")
    parser.add_argument("--channel_independence", type=bool, default=False, help="compatibility argument")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)
    parser.add_argument("--class_strategy", type=str, default="projection", help="compatibility argument")
    parser.add_argument("--target_root_path", type=str, default="./data/electricity/", help="root path of the data file")
    parser.add_argument("--target_data_path", type=str, default="electricity.csv", help="data file")
    parser.add_argument("--efficient_training", type=bool, default=False, help="compatibility argument")
    parser.add_argument("--partial_start_index", type=int, default=0, help="compatibility argument")

    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument(
        "--skip_test_after_train",
        type=str2bool,
        default=False,
        help="skip the automatic test phase after Stage 1 training",
    )

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_Long_Term_Forecast_DLinear

    if args.is_training:
        for ii in range(args.itr):
            setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.model_id,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy,
                ii,
            )

            exp = Exp(args)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            if not args.skip_test_after_train:
                print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.test(setting)
            else:
                print(f">>>>>>>skip testing after training : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            if args.do_predict:
                print(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.model_id,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy,
            ii,
        )

        exp = Exp(args)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
