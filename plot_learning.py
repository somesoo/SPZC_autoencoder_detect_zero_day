import matplotlib.pyplot as plt
import re

# Wczytaj logi z pliku
with open("train.log", "r") as file:
    lines = file.readlines()

losses_all_trials = []
current_losses = []

for line in lines:
    trial_start = re.match(r"^=== Próba \d+/\d+ ===", line)
    loss_line = re.match(r"\[\d+/\d+\] Train Loss: ([0-9.]+)", line)

    if trial_start:
        if current_losses:
            losses_all_trials.append(current_losses)
            current_losses = []

    elif loss_line:
        current_losses.append(float(loss_line.group(1)))

# Dodaj ostatnią próbę
if current_losses:
    losses_all_trials.append(current_losses)

# Sprawdzenie
assert len(losses_all_trials) == 24, f"Oczekiwano 24 prób, znaleziono: {len(losses_all_trials)}"

# Rysowanie wspólnego wykresu
plt.figure(figsize=(12, 6))

for i, loss_values in enumerate(losses_all_trials):
    plt.plot(range(1, len(loss_values) + 1), loss_values, label=f'Próba {i+1}', linewidth=1)

plt.title("Train Loss dla 24 prób (autoenkoder)", fontsize=14)
plt.xlabel("Epoka", fontsize=12)
plt.ylabel("Train Loss (MSE)", fontsize=12)
plt.grid(True)
plt.legend(fontsize=8, ncol=4, loc='upper right')
plt.tight_layout()
plt.savefig("combined_loss_plot.png", dpi=300)
plt.show()