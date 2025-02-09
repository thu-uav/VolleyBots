import datetime
import logging
import os
import urllib.request

import pytz
import wandb
from omegaconf import DictConfig, OmegaConf


def dict_flatten(a: dict, delim="."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg: DictConfig, *run_name_args):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg: DictConfig = cfg.wandb

    try:
        response = urllib.request.urlopen("http://www.baidu.com", timeout=1)
        timestamp = response.headers["Date"]
        gmt_time = datetime.datetime.strptime(timestamp, "%a, %d %b %Y %H:%M:%S %Z")
        gmt_time = gmt_time.replace(tzinfo=pytz.timezone("GMT"))
        cst_tz = pytz.timezone("Asia/Shanghai")
        cst_time = gmt_time.astimezone(cst_tz)
        time_str = cst_time.strftime("%m-%d_%H-%M")
        print("Beijing time:", time_str)
    except Exception as e:
        time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
        print("Failed to get Beijing time. Use local time instead.")
        print("Local time:", time_str)

    run_name = f"{wandb_cfg.run_name}/{cfg.task.drone_model}/{time_str}"
    for arg in run_name_args:
        if arg == "":
            continue
        run_name += f"/{arg}"
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=run_name,
        mode=wandb_cfg.get("mode", "disabled"),
        tags=wandb_cfg.tags,
    )
    if wandb_cfg.run_id is not None:
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
    else:
        kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict)
    return run
