This is the training code used to train [StripedHyena-Nous-7B](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B).

First, tokenize your data

```sh
python tokenization.py \
    --dataset your-super-cool-sharegpt-format-dataset \
    --tokenizer togethercomputer/StripedHyena-Hessian-7B \
    --output tokenized \
    --num-proc 32 \
    --pad-to-length 4096 \
    --truncate
```

Make sure you have done `accelerate config` -- we used the provided DeepSpeed configuration.
Then, train!

```sh
accelerate launch finetune.py \
    --model togethercomputer/StripedHyena-Hessian-7B \
    --dataset tokenized \
    --output-dir output \
    --epochs 4 \
    --batch-size 12 \
    --gradient-accumulate-every 12 \
    --warmup-steps 350 \
    --learning-rate 0.000004 \
    --lr-schedule linear \
    --weight-decay 0.1 \
    --checkpointing-steps 1000 \
    --no-decay poles residues
```

The `--no-decay` option disables weight decay on *only* the specified parameters.
For StripedHyena, we've found that disabling weight decay on the Hyena operator's `poles` and `residues` parameters improves performance.
There is also an option `--frozen` that can completely freeze select parameter groups.
