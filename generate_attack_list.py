import os

FOLDER = "separated"
OUTPUT = "attack_list.txt"

attacks = set()

for fname in os.listdir(FOLDER):
    if not fname.endswith(".csv"):
        continue
    if fname.lower().startswith("benign"):
        continue
    if "__" in fname:
        attack_name = fname.split("__")[0].lower()
        attacks.add(attack_name)

with open(OUTPUT, "w") as f:
    for name in sorted(attacks):
        f.write(name + "\n")

print(f"Zapisano {len(attacks)} unikalnych ataków do: {OUTPUT}")
