import argparse
import json
import os
import socket
import time

import jax
import jax.numpy as jnp

from glassogp import GroupLassoGaussianProcess
from inclinerunning import InclineRunning

jax.config.update("jax_enable_x64", True)
print("JAX devices:", jax.devices())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/Incline Running")
    parser.add_argument("--subsample", type=int, default=20)
    parser.add_argument("--feet", default="both", choices=["both", "left_only", "right_only"])
    parser.add_argument("--target", default="markers", choices=["markers", "forces"])
    parser.add_argument("--inclines", default="inc0", choices=["all", "inc0", "inc5", "inc10"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--adapt_rho", action="store_true", default=False)
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--lambda_val", type=float, default=0.0)
    parser.add_argument("--output_dir", default="results/bench")
    args = parser.parse_args()

    group_size = 6 if args.feet == "both" else 3
    autoregressive = args.target == "markers"

    data = InclineRunning(
        path=args.data_dir,
        subsample=args.subsample,
        feet=args.feet,
        target=args.target,
        inclines=args.inclines,
    )
    x_train = jnp.asarray(data.x_train)
    y_train = jnp.asarray(data.y_train)

    print(f"Data: {x_train.shape[0]} train samples, {x_train.shape[1]} features, {y_train.shape[1]} outputs")
    print(f"Config: adapt_rho={args.adapt_rho}, n_jobs={args.n_jobs}, lambda={args.lambda_val}, n_iters={args.n_iters}")

    def fit(n_iters):
        return GroupLassoGaussianProcess.fit(
            x_train=x_train,
            y_train=y_train,
            l1_penalty=jnp.array(args.lambda_val),
            group_size=group_size,
            autoregressive=autoregressive,
            max_iterations=n_iters,
            tol=jnp.array(-jnp.inf),  # never early-stop; always run exactly n_iters
            adapt_rho=args.adapt_rho,
            n_jobs=args.n_jobs,
        )

    # warmup: 1 iteration to trigger JIT compilation
    print("Warming up JIT...")
    fit(1)

    print("Timing...")
    t0 = time.perf_counter()
    fit(args.n_iters)
    wall_time = time.perf_counter() - t0

    devices = jax.devices()
    result = {
        "hostname": socket.gethostname(),
        "device": str(devices[0]),
        "device_platform": devices[0].platform,
        "n_jobs": args.n_jobs,
        "adapt_rho": args.adapt_rho,
        "n_iters": args.n_iters,
        "wall_time_s": round(wall_time, 3),
        "time_per_iter_s": round(wall_time / args.n_iters, 3),
        "n_train": int(x_train.shape[0]),
        "n_features": int(x_train.shape[1]),
        "n_outputs": int(y_train.shape[1]),
        "subsample": args.subsample,
        "lambda_val": args.lambda_val,
    }

    print(json.dumps(result, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"adapt{int(args.adapt_rho)}_{devices[0].platform}_{socket.gethostname()}"
    out_path = os.path.join(args.output_dir, f"bench_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
