"""
Figure 6: reconstruction error over time for a single (M13, M7, M3) run,
shown for the three reconstruction pathways.

Requires a completed run named `13_7_3_seed1` (see run_experiments.sh).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = '13_7_3_seed1'
max_step = 200
d = json.load(open(f'particle_visualizations/{run}/reconstruction_errors.json'))
steps = np.array(d['iterations'])
mask = steps <= max_step

series = [
    ('13->7->13',       r'$M_{13}\to M_7 \to M_{13}$',                 '#1f77b4'),
    ('7->3->7',         r'$M_7 \to M_3 \to M_7$',                      '#ff7f0e'),
    ('13->7->3->7->13', r'$M_{13}\to M_7 \to M_3 \to M_7 \to M_{13}$', '#2ca02c'),
]

fig, ax = plt.subplots(figsize=(7, 4.5))
for key, label, color in series:
    ax.plot(steps[mask], np.array(d['errors'][key])[mask], color=color, lw=1.6,
            marker='o', markersize=4, label=label)

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Reconstruction Error', fontsize=12)
ax.set_xlim(0, max_step)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=13)
plt.tight_layout()
out = 'figures/fig6_reconstruction_error.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
