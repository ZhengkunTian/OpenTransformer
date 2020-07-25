'''
@Author: Zhengkun Tian
@Email: zhengkun.tian@outlook.com
@Date: 2020-04-16 21:50:10
@LastEditTime: 2020-04-16 21:52:56
@FilePath: \OpenTransformer\tools\average.py
'''
import torch
import os
import sys

rootdir = sys.argv[1]
st = sys.argv[2]
ed = sys.argv[3]


def average_chkpt(datadir, start, end):
    id_chkpt = [str(i) for i in range(int(start), int(end)+1)]
    print('Average these number %s models' % ','.join(id_chkpt))

    chkpts = ['model.epoch.%s.pt' % idx for idx in id_chkpt]

    params_dict = {}
    params_keys = {}
    new_state = None
    num_models = len(chkpts)

    for chkpt in chkpts:
        state = torch.load(os.path.join(datadir, chkpt))
        # Copies over the settings from the first checkpoint

        if new_state is None:
            new_state = state

        for key, value in state.items():

            if key != 'model': continue

            model_params = value
            model_params_keys = list(model_params.keys())

            if key not in params_keys:
                params_keys[key] = model_params_keys

            if key not in params_dict:
                params_dict[key] = {}

            for k in params_keys[key]:
                p = model_params[k]
                # if isinstance(p, torch.HalfTensor)
                #     p = p.float()

                if k not in params_dict[key]:
                    params_dict[key][k] = p.clone()
                    # NOTE: clone() is needed in case of p is a shared parameter
                else:
                    params_dict[key][k] += p

    averaged_params = {}
    for key, states in params_dict.items():
        averaged_params[key] = {}
        for k, v in states.items():
            averaged_params[key][k] = v
            averaged_params[key][k].div_(num_models)
    
        new_state[key] = averaged_params[key]

    torch.save(new_state, os.path.join(datadir, 'model.average.from%sto%s.pt' % (start, end)))
    print('Save the average checkpoint as %s' % os.path.join(datadir,  'model.average.from%sto%s.pt' % (start, end)))
    print('Done!')


average_chkpt(rootdir, st, ed)
