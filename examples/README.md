# Приклад даних варіанта

`swarm-sim scenarios --variant N`, `swarm-sim view --variant N` і `simulator.scenario_set(N)` шукають теку
`variant_NN/student/` (або `data/variant_NN/student/`) у поточній теці чи в батьківській:

- `variant.json` — розподіл стартових умов варіанта: діапазон і розподіл кількості дронів (`n_lo`, `n_hi`,
  `n_shape`: «рівномірний» | «більше малих» | «більше великих»), вітер (`wind_max`), пориви (`gust_min`, `gust_max`),
  відстань і висота цілі (`goal_dist`, `goal_z`), стартовий куб (`cube`), початкова швидкість (`v0_max`),
  `sigma_y` — масштаб online MSE (σ дій у body / world).
- `eval_scenarios.json`, `stress_scenarios.json` — готові набори (`--set eval` / `--set stress`); їх створює
  `swarm-sim scenarios --variant N --seed S --n K -o …` або `simulator.scenarios.save_scenarios`.

```bash
cd examples
swarm-sim scenarios --variant 1 --seed 7001 --n 30 --n-agents 5 -o variant_01/student/eval_scenarios.json
swarm-sim fly --model reynolds --scenarios variant_01/student/eval_scenarios.json -o rec.parquet
swarm-sim metrics rec.parquet
swarm-sim view --model reynolds --variant 1
```
