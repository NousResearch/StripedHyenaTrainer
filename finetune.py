import argparse
import copy
import os
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, set_seed as transformers_set_seed, \
    default_data_collator, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


def main(args):

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.login()

    set_seed(args.seed)
    transformers_set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout]
    )
    accelerator.init_trackers(project_name=args.wandb)
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        config=config,
        use_flash_attention_2=True,
        trust_remote_code=True
    )

    if args.print_parameter_dtypes and accelerator.is_main_process:
        for n, p in model.named_parameters():
            print(f"{n}: {p.dtype}")

    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    if "labels" not in train_dataset.column_names:
        def add_labels(sample):
            sample["labels"] = copy.deepcopy(sample["input_ids"])
            return sample
        train_dataset = train_dataset.map(
            add_labels, desc="Adding labels", num_proc=args.num_proc)
    if "attention_mask" not in train_dataset.column_names:
        def add_attention_mask(sample):
            sample["attention_mask"] = torch.ones(
                len(sample["input_ids"]), dtype=torch.int8)
            return sample
        train_dataset = train_dataset.map(
            add_attention_mask, desc="Adding attention mask", num_proc=args.num_proc)

    if args.truncate:
        def truncate(sample):
            sample["input_ids"] = sample["input_ids"][0:args.truncate]
            sample["labels"] = sample["labels"][0:args.truncate]
            sample["attention_mask"] = sample["attention_mask"][0:args.truncate]
            return sample
        train_dataset = train_dataset.map(
            truncate, desc="Truncating", num_proc=args.num_proc)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size
    )

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulate_every
    epoch_size = args.max_train_steps if args.max_train_steps else len(train_dataset)
    num_training_steps = args.max_train_steps if args.max_train_steps else epoch_size * args.epochs // total_batch_size
    num_epoch_steps = num_training_steps // args.epochs
    accelerator.print(f"Total batch size: {total_batch_size}")
    accelerator.print(f"Epoch training steps: {num_epoch_steps}")
    accelerator.print(f"Total training steps: {num_training_steps} ({args.epochs} epochs)")

    param_optimizer = list(model.named_parameters())
    frozen = args.frozen[0] if args.frozen else []
    frozen_parameter_names = [n for n, p in param_optimizer if any(nd in n for nd in frozen)]
    if len(frozen_parameter_names) > 0:
        accelerator.print(f"Frozen parameters: {frozen_parameter_names}")
    for n, p in param_optimizer:
        if any(nd in n for nd in frozen):
            p.requires_grad = False

    no_decay = args.no_decay[0] if args.no_decay else []
    if len(no_decay) > 0:
        undecayed_parameter_names = [n for n, p in param_optimizer if any(nd in n for nd in no_decay)]
        if len(undecayed_parameter_names) > 0:
            accelerator.print(f"Undecayed parameters: {undecayed_parameter_names}")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay if args.weight_decay else 0.0,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
    else:
        optimizer_grouped_parameters = list(model.parameters())

    warmup_steps = args.warmup_steps
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        warmup_steps *= accelerator.num_processes

    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        model = accelerator.prepare(model)
    optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                                betas=(args.adamw_beta1, args.adamw_beta2), eps=args.adamw_eps)
    if args.lr_schedule == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optim, num_training_steps=num_training_steps, num_warmup_steps=warmup_steps)
    elif args.lr_schedule == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps)
    elif args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optim, num_training_steps=num_training_steps, num_warmup_steps=warmup_steps)
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        optim, train_loader, scheduler = accelerator.prepare(
            optim, train_loader, scheduler)
    else:
        model, optim, train_loader, scheduler = accelerator.prepare(
            model, optim, train_loader, scheduler)

    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.distributed_type == DistributedType.FSDP:
        model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    progress_bar = tqdm(
        range(num_training_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        resume_step = (
            int(training_difference.replace("step_", ""))
        )

    if args.resume_from_checkpoint and resume_step is not None:
        train_loader = accelerator.skip_first_batches(
            train_loader, resume_step % num_epoch_steps)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")
        epoch = resume_step // num_epoch_steps
    else:
        epoch = 0

    loss_file = open(args.log_loss, "a" if args.resume_from_checkpoint else "w", encoding="utf-8") if args.log_loss and accelerator.is_main_process else None

    if not args.save_only:
        model.train()
        for _ in range(epoch, args.epochs):
            for _, batch in enumerate(train_loader):
                loss_log = None
                with accelerator.accumulate(model):
                    loss = model(**batch).loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        last_lr = scheduler.get_last_lr()
                        if isinstance(last_lr, list):
                            last_lr = last_lr[0]
                        loss_log = {
                            "loss": loss.item(),
                            "epoch": completed_steps / num_epoch_steps,
                            "learning_rate": last_lr
                        }
                        accelerator.log(loss_log, step=completed_steps)
                        if loss_file is not None:
                            loss_file.write(f"{loss_log['loss']},")
                            loss_file.flush()

                        if isinstance(args.grad_norm, float):
                            accelerator.clip_grad_norm_(
                                model.parameters(), args.grad_norm)
                        
                    optim.step()
                    scheduler.step()
                    optim.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if loss_log is not None:
                        progress_bar.set_postfix(loss_log)
                    completed_steps += 1

                    if isinstance(args.checkpointing_steps, int) and completed_steps > 0:
                        if completed_steps % args.checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(
                                    args.output_dir, output_dir)
                            accelerator.save_state(output_dir)
                if completed_steps >= num_training_steps:
                    break
            if completed_steps >= num_training_steps:
                break

        accelerator.print("Training Finished")
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        accelerator.wait_for_everyone()

        if accelerator.distributed_type == DistributedType.FSDP:
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = accelerator.get_state_dict(model, unwrap=False)
        else:
            state_dict = accelerator.get_state_dict(model)

        accelerator.unwrap_model(model).save_pretrained(
            f"{args.output_dir}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        accelerator.print("Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--resume-from-checkpoint", type=str)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--warmup-steps", type=int, default=20)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--grad-norm", type=float)
    args.add_argument("--model", type=str, required=True)
    args.add_argument("--truncate", type=int)
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--num-proc", type=int, default=32)
    args.add_argument("--lr-schedule", type=str,
                      choices=["linear", "constant", "cosine"], default="linear")
    args.add_argument("--weight-decay", type=float)
    args.add_argument("--no-decay", action="append", nargs="+")
    args.add_argument("--frozen", action="append", nargs="+")
    args.add_argument("--adamw-beta1", type=float, default=0.9)
    args.add_argument("--adamw-beta2", type=float, default=0.999)
    args.add_argument("--adamw-eps", type=float, default=1e-8)
    args.add_argument("--save-only", action="store_true")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--print-parameter-dtypes", action="store_true")
    main(args.parse_args())
