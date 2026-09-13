import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

import sys
sys.path.append("/workspace/tiny-llm-from-scratch")

from tokenizer.char_tokenizer import CharTokenizer
from model.tiny_transformer import GPTConfig, TinyGPT, count_parameters


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters):
    model.eval()
    out = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_100k.json")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer_char.json")
    parser.add_argument("--train", default="data/train.txt")
    parser.add_argument("--val", default="data/val.txt")
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    torch.set_num_threads(4)

    cfg = load_json(args.config)
    tokenizer = CharTokenizer.load(args.tokenizer)

    device = cfg.get("device", "cpu")

    train_text = Path(args.train).read_text(encoding="utf-8")
    val_text = Path(args.val).read_text(encoding="utf-8")

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
    )

    model = TinyGPT(gpt_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    param_count = count_parameters(model)

    print("Model name:", cfg["model_name"])
    print("Device:", device)
    print("Vocab size:", tokenizer.vocab_size)
    print("Parameter count:", param_count)
    print("Train tokens:", len(train_ids))
    print("Val tokens:", len(val_ids))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path("logs") / f"{cfg['model_name']}_training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_time = time.time()

    for step in tqdm(range(cfg["max_steps"])):
        if step % cfg["eval_interval"] == 0:
            losses = estimate_loss(
                model,
                train_ids,
                val_ids,
                cfg["batch_size"],
                cfg["block_size"],
                device,
                cfg["eval_iters"],
            )

            elapsed_min = (time.time() - start_time) / 60

            line = (
                f"step={step} "
                f"train_loss={losses['train']:.4f} "
                f"val_loss={losses['val']:.4f} "
                f"elapsed_min={elapsed_min:.2f}"
            )

            print(line)

            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                checkpoint_path = out_dir / f"{cfg['model_name']}_best.pt"

                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "gpt_config": gpt_config.__dict__,
                    "tokenizer_path": args.tokenizer,
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "param_count": param_count,
                }, checkpoint_path)

                print("Saved best checkpoint:", checkpoint_path)

        xb, yb = get_batch(
            train_ids,
            cfg["batch_size"],
            cfg["block_size"],
            device
        )

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    final_path = out_dir / f"{cfg['model_name']}_final.pt"

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "gpt_config": gpt_config.__dict__,
        "tokenizer_path": args.tokenizer,
        "step": cfg["max_steps"],
        "best_val_loss": best_val_loss,
        "param_count": param_count,
    }, final_path)

    print("Training completed.")
    print("Final checkpoint:", final_path)
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
