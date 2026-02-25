import json
import matplotlib.pyplot as plt

log_path = "/home/tianle/dinolink_project/outputs/nuSenes2d_dinolink_wo_VQ/log.txt"

epochs = []
train_loss = []
test_loss = []

with open(log_path, "r") as f:
    for i, line in enumerate(f, 1):
        d = json.loads(line)
        epochs.append(i)
        train_loss.append(d["train_loss"])
        if "test_loss" in d:
            test_loss.append(d["test_loss"])
        else:
            test_loss.append(None)  # 没有就占位

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="train_loss")
# 只画有值的 test_loss
test_epochs = [e for e, v in zip(epochs, test_loss) if v is not None]
test_vals   = [v for v in test_loss if v is not None]
if test_epochs:
    plt.plot(test_epochs, test_vals, label="test_loss")

plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("nuSenes2d_dinolink_wo_VQ convergence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/tianle/dinolink_project/outputs/nuSenes2d_dinolink_wo_VQ/convergence.png")