"""
Figure 7: full-path reconstruction error (individual seeds) for two converging
presets, (M13, M7, M3) and (M13, M5, M3), versus a fixed-weights control.

Shows that (i) hierarchical compression reproduces across molecule-length
combinations, and (ii) it requires the state->catalyst feedback: with fixed
weights the error stays high.

Requires completed runs `13_7_3_seed{1..5}`, `13_5_3_seed{1..5}`,
`fixed_weights_seed{1..5}` (see run_experiments.sh).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

max_step = 1000
conditions = [
    ('13_7_3',        r'$(M_{13}, M_7, M_3)$',                '13->7->3->7->13', '#2c6fbb'),
    ('13_5_3',        r'$(M_{13}, M_5, M_3)$',                '13->5->3->5->13', '#27ae60'),
    ('fixed_weights', r'$(M_{13}, M_7, M_3)$ fixed weights',  '13->7->3->7->13', '#c0392b'),
]
seeds = [1, 2, 3, 4, 5]

fig, ax = plt.subplots(figsize=(7.5, 5))

for prefix, label, key, color in conditions:
    first = True
    for s in seeds:
        p = f'particle_visualizations/{prefix}_seed{s}/reconstruction_errors.json'
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if key not in d['errors']:
            continue
        steps = np.array(d['iterations'])
        vals = np.array(d['errors'][key])
        m = steps <= max_step
        ax.plot(steps[m], vals[m], color=color, lw=1.3, alpha=0.8,
                label=label if first else None)
        first = False

ax.set_xlabel('Simulation step', fontsize=15)
ax.set_ylabel('Reconstruction Error (MSE)', fontsize=15)
ax.set_xlim(0, max_step)
ax.set_ylim(bottom=0)
ax.tick_params(axis='both', labelsize=13)
ax.grid(True, alpha=0.3, ls='--')
ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.16),
          ncol=3, frameon=False)
plt.tight_layout()
out = 'figures/fig7_preset_control_comparison.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
