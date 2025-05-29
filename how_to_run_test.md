python3 run_test.py \
  --model models/best_model_trial_03_drop0.1_bs512_lr0.001_bnTrue.pt \
  --test-type single \
  --attack-files separated/DDoS__Friday-WorkingHours-Afternoon.csv \
  --thresholds 0.01 0.02 0.03


python3 run_test.py \
  --model models/best_model_....pt \
  --test-type mix \
  --attack-files separated/DDoS__*.csv separated/PortScan__*.csv \
  --thresholds 0.02 0.04 0.06


python3 run_test.py \
  --model models/best_model_....pt \
  --test-type benign \
  --thresholds 0.01 0.02 0.03 0.04



Każde wywołanie tworzy w ./tests/<timestamp>_<test-type>/:
    X_test.npy, y_test.npy – dokładnie te dane, które były eval.
    params.json – parametry wywołania.
    metrics.json i metrics_table.csv – metryki dla wszystkich thresholdów.