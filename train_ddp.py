#!/usr/bin/env python
"""PyTorch DDP 학습: data/synthetic.npz -> runs/<GPU 수>gpu/{metrics.csv, model.pt, train.log}"""
import argparse
import csv
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torchmetrics import MeanSquaredError


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic.npz")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--global-batch", type=int, default=32768, help="GPU당 배치 = global-batch / GPU 수")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--layers", type=int, default=6)
    return p.parse_args()


def build_model(hidden=2048, layers=6):
    mods, d = [], 1
    for _ in range(layers):
        mods += [torch.nn.Linear(d, hidden), torch.nn.SiLU()]
        d = hidden
    mods.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*mods)


def main():
    args = get_args()

    # torchrun 이 설정한 환경 변수
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(0)
    run_dir = Path("runs") / f"{world}gpu"
    if rank == 0:   # rank 0 만 출력·기록·저장
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(run_dir / "train.log", "w")

    def log(msg):   # 화면 + runs/<N>gpu/train.log
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

    # rank 배치 확인
    me = (rank, socket.gethostname().split(".")[0], local)
    everyone = [None] * world
    dist.all_gather_object(everyone, me)
    if rank == 0:
        log(f"world_size={world}, nodes={sorted({host for _, host, _ in everyone})}")
        for r, host, gpu in everyone:
            log(f"  rank {r} = {host} GPU{gpu}")

    # 데이터 전체를 GPU 에 적재, DistributedSampler 로 rank 별 1/N 분할
    batch = args.global_batch // world
    d = np.load(args.data)
    x_train, y_train = (torch.from_numpy(d[k]).to(device) for k in ("x_train", "y_train"))
    x_val, y_val = (torch.from_numpy(d[k]).to(device) for k in ("x_val", "y_val"))
    train_sampler = DistributedSampler(range(len(x_train)), shuffle=True, drop_last=True)
    val_idx = torch.tensor(list(DistributedSampler(range(len(x_val)), shuffle=False)), device=device)
    steps = len(train_sampler) // batch

    # backward 시 gradient all-reduce
    model = DDP(build_model(args.hidden, args.layers).to(device), device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.MSELoss()
    # compute() 시 rank 간 동기화
    train_mse, val_mse = MeanSquaredError().to(device), MeanSquaredError().to(device)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        log(f"model {n / 1e6:.1f}M params, batch {batch}/GPU x {world} GPU = {batch * world}, "
              f"{steps} steps/epoch")
        log(f"{'epoch':>5} {'train_loss':>10} {'val_loss':>9} {'time(s)':>8} {'samples/s':>11}")

    rows = []
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)   # 에폭별 셔플
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx = torch.tensor(list(train_sampler), device=device)
        for i in range(steps):
            b = idx[i * batch:(i + 1) * batch]
            x, y = x_train[b], y_train[b]
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_mse.update(pred.detach(), y)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sched.step()

        model.eval()
        with torch.no_grad():
            for b in val_idx.split(batch):
                val_mse.update(model(x_val[b]), y_val[b])
        train_loss, val_loss = train_mse.compute().item(), val_mse.compute().item()
        train_mse.reset(); val_mse.reset()

        if rank == 0:
            sps = steps * batch * world / dt
            log(f"{epoch:>5} {train_loss:>10.5f} {val_loss:>9.5f} {dt:>8.2f} {sps:>11,.0f}")
            rows.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                             epoch_time_s=dt, samples_per_s=sps, gpus=world))
            with open(run_dir / "metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)

    if rank == 0:
        torch.save(model.module.state_dict(), run_dir / "model.pt")   # DDP wrapper 제외
        log(f"done -> {run_dir}/metrics.csv, {run_dir}/model.pt, {run_dir}/train.log")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
