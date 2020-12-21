import os
import glob
import torch
import smtplib
import logging
from email.mime.text import MIMEText


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


def sendEmail(title, content, receivers=['zhengkun.tian@outlook.com']):

    mail_host = 'smtp.163.com'  
    mail_user = 'betterdamon'  
    mail_pass = 'YXBUQMJHBDIUMXBN'
    sender = 'betterdamon@163.com'  

    message = MIMEText(content, 'plain', 'utf-8')  # 内容, 格式, 编码
    message['From'] = "{}".format(sender)
    message['To'] = ",".join(receivers)
    message['Subject'] = title
 
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)  # 启用SSL发信, 端口一般是465
        smtpObj.login(mail_user, mail_pass)  # 登录验证
        smtpObj.sendmail(sender, receivers, message.as_string())  # 发送
        print("mail has been send successfully.")
    except smtplib.SMTPException as e:
        print(e)


def average_parameters(expdir, N=20):

    chkpts = glob.glob(os.path.join(expdir, 'model.epoch.*.pt'))
    assert len(chkpts) >= N

    sorted_chkpts = sorted(chkpts, key=lambda x: int(x.split('.')[-2]), reverse=False)
    last_n_chkpts = sorted_chkpts[-N:]

    params_dict = {}
    params_keys = {}
    new_state = None

    for chkpt in last_n_chkpts:
        state = torch.load(chkpt)
        # Copies over the settings from the first checkpoint

        if new_state is None:
            new_state = state

        for key, value in state.items():

            if key in ['params', 'epochs', 'amp', 'global_step']: continue

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
            averaged_params[key][k].div_(N)
    
        new_state[key] = averaged_params[key]

    save_chkpt_path = os.path.join(expdir, 'model.average.last.%d.pt' % N)
    torch.save(new_state, save_chkpt_path)
    print('Save the average checkpoint as %s' % save_chkpt_path)

    return save_chkpt_path


def count_parameters(named_parameters):
    # Count total parameters
    total_params = 0
    part_params = {}
    for name, p in sorted(list(named_parameters)):
        n_params = p.numel()
        total_params += n_params
        logging.debug("%s %d" % (name, n_params))
        part_name = name.split('.')[0]
        if part_name in part_params:
            part_params[part_name] += n_params
        else:
            part_params[part_name] = n_params
    
    for name, n_params in part_params.items():
        logging.info('%s #params: %d' % (name, n_params))
    logging.info("Total %.2f M parameters" % (total_params / 1000000))
    logging.info('Estimated Total Size (MB): %0.2f' % (total_params * 4. /(1024 ** 2)))