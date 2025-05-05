import os
import os.path as op
from logging import warning

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

try:
    from deeptaskgen.datasets.taskgen_dataset import load_timeseries
    from deeptaskgen.utils.parser import default_parser
except ImportError:
    import sys

    path = op.abspath(op.join(op.dirname(__file__), ".."))
    if path not in sys.path:
        sys.path.append(path)
    del sys, path
    from deeptaskgen.datasets.taskgen_dataset import load_timeseries
    from deeptaskgen.utils.parser import default_parser


def predict(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not op.exists(args.working_dir):
        os.makedirs(args.working_dir)
    else:
        warning(
            f"{args.working_dir} already exists!, The existing files will not be overwritten!"
        )

    """Load Datalist"""
    subj_ids = np.genfromtxt(args.test_list, dtype=int, delimiter=",")
    if subj_ids.ndim == 0:
        # If only one subject ID is provided, convert it to a list
        subj_ids = np.array([subj_ids])
    if subj_ids[0] == -1:
        subj_ids = np.genfromtxt(args.test_list, dtype=str, delimiter=",")

    """Init Model"""
    if args.checkpoint_file is None:
        raise ValueError("Must provide a checkpoint to load!")
    model = args.architecture.load_from_checkpoint(
        args.checkpoint_file,
        in_chans=args.n_channels,
        out_chans=args.n_out_channels,
        fdim=args.fdim,
        activation=args.activation,
        optimizer=args.optimizer,
        up_mode=args.upsampling_mode,
        loss_fn=args.loss,
        add_loss=args.add_loss,
        max_level=args.max_depth,
        n_conv=args.n_conv_layers,
        batch_norm=args.batch_norm,
        lr_scheduler=args.lr_scheduler,
    ).to(args.device)

    # Masking
    if args.pred_mask is not None:
        pred_mask = torch.from_numpy(nib.load(args.pred_mask).get_fdata()).to(
            args.device
        )
    else:
        pred_mask = None

    """Predict"""
    print("Predicting...")
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(subj_ids)) as pbar:
            for id in subj_ids:
                pbar.set_description(f"Predicting {id}...")
                pbar.update(1)
                pred_file = op.join(args.working_dir, f"{id}_pred.npy")
                if not op.exists(pred_file):
                    pred_list = []
                    for sample_id in range(args.n_samples_per_subj):
                        rest_file = op.join(
                            args.rest_dir, f"{id}_sample{sample_id}_rsfc.npy"
                        )
                        img = load_timeseries(
                            rest_file,
                            mask=args.mask,
                            unmask=args.unmask,
                            device=args.device,
                        )
                        if pred_mask is not None:
                            img = img * pred_mask
                        pred_list.append(
                            model(img.unsqueeze(0)).cpu().detach().numpy().squeeze(0)
                        )
                    np.save(pred_file, np.array(pred_list))
                else:
                    warning(f"Skipping {id} because prediction already exists!")

    print("Finished predicting!")


if __name__ == "__main__":
    predict(default_parser())
