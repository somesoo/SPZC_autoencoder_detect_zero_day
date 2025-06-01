import matplotlib.pyplot as plt
import re
import ast

# Wczytaj logi z pliku
with open("train.log", "r") as file:
    lines = file.readlines()


losses_all_trials = []
trial_labels = []
current_losses = []

for line in lines:
    trial_start = re.match(r"^=== Próba (\d+)/\d+ ===", line)
    param_line = re.match(r"^Parametry: ({.*})", line)
    loss_line = re.match(r"\[\d+/\d+\] Train Loss: ([0-9.]+)", line)

    if trial_start:
        if current_losses:
            losses_all_trials.append(current_losses)
            current_losses = []

    elif param_line:
        param_dict = ast.literal_eval(param_line.group(1))
        # Tworzymy etykietę w formacie: NR, batch, batchnorm, dropout, lr
        nr = len(losses_all_trials) + 1
        label = f"{nr}, {param_dict['batch_size']}, {param_dict['batchnorm']}, {param_dict['dropout']}, {param_dict['lr']}"
        trial_labels.append(label)

    elif loss_line:
        current_losses.append(float(loss_line.group(1)))

# Dodaj ostatnią próbę
if current_losses:
    losses_all_trials.append(current_losses)

# Sprawdź poprawność
assert len(losses_all_trials) == len(trial_labels) == 20, \
    f"Oczekiwano 24 prób, znaleziono: {len(losses_all_trials)}, etykiety: {len(trial_labels)}"

# Rysowanie wykresu
plt.figure(figsize=(14, 7))

for loss_values, label in zip(losses_all_trials, trial_labels):
    plt.plot(range(1, len(loss_values)+1), loss_values, label=label, linewidth=1)

plt.title("Train Loss dla 20 prób (autoenkoder)", fontsize=14)
plt.xlabel("Epoka", fontsize=12)
plt.ylabel("Train Loss (MSE)", fontsize=12)
plt.grid(True)
plt.legend(title="NR, batch, batchnorm, dropout, lr", title_fontsize=10, fontsize=8, ncol=2, loc='upper right')
plt.tight_layout()
plt.savefig("combined_loss_with_params.png", dpi=300)
plt.show()