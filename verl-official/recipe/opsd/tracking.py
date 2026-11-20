"""Rank-zero W&B tracking for OPSD; training data and model weights stay local."""

import json
from pathlib import Path


def start_run(args, output, world):
    """Resume the collection run, or create one for a directly launched training job."""
    import wandb

    config = dict(vars(args), world_size=world, micro_batch_size_per_gpu=1,
                  global_batch_size=world * args.accumulation_steps,
                  effective_gradient_accumulation_steps=args.accumulation_steps)
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
        id=args.wandb_run_id, resume="allow" if args.wandb_run_id else None,
        mode=args.wandb_mode, dir=str(output), config=config,
        settings=wandb.Settings(init_timeout=120),
    )
    # A custom x-axis lets collection and training share one run without step conflicts.
    run.define_metric("train/step")
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("env/*", step_metric="train/step")
    run.define_metric("actor/entropy_loss", step_metric="train/step")
    run.define_metric("validation/*", step_metric="train/step")
    Path(output, "wandb_run.json").write_text(json.dumps({"id": run.id, "url": run.url}, indent=2))
    return run
