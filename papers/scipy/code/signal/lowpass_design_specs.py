from __future__ import division, print_function

import matplotlib.pyplot as plt


deltap = 0.13
deltas = 0.15

omegac = 0.4
delta_omega = 0.1

plt.figure(figsize=(4.0, 2.5))

plt.plot([0, omegac - 0.5*delta_omega], [1 + deltap, 1 + deltap], 'k',
         linewidth=1)
plt.plot([0, omegac - 0.5*delta_omega], [1 - deltap, 1 - deltap], 'k',
         linewidth=1)
plt.plot([0, omegac - 0.5*delta_omega], [1, 1], 'k--', alpha=0.5,
         linewidth=1)
plt.plot([omegac + 0.5*delta_omega, 1], [deltas, deltas], 'k',
         linewidth=1)


plt.xlim(0, 1)
ax = plt.gca()
ax.set_xticks([0, omegac, 1])
ax.set_xticklabels(['0', r'$\omega_c$', r'$\pi$'])
ax.set_yticks([0, deltas, 1-deltap, 1, 1+deltap])
ax.set_yticklabels(['0', r'$\delta_s$', r'$1 - \delta_p$', '1',
                    r'$1 + \delta_p$'])
plt.ylim(0, 1.2)

plt.axvline(omegac - 0.5*delta_omega, color='k', linewidth=1, alpha=0.15)
plt.axvline(omegac + 0.5*delta_omega, color='k', linewidth=1, alpha=0.15)

rect = plt.Rectangle((0, 1 + deltap),
                     omegac - 0.5*delta_omega, 0.1,
                     facecolor='k',
                     edgecolor='k',
                     alpha=0.15)
ax.add_patch(rect)
rect = plt.Rectangle((0, 1 - deltap),
                     omegac - 0.5*delta_omega, -1,
                     facecolor='k',
                     edgecolor='k',
                     alpha=0.15)
ax.add_patch(rect)
rect = plt.Rectangle((omegac + 0.5*delta_omega, deltas),
                     1 - omegac - 0.5*delta_omega, 1.2,
                     facecolor='k',
                     edgecolor='k',
                     alpha=0.15)
ax.add_patch(rect)

plt.annotate(r'$\Delta\omega$', (omegac+0.5*delta_omega, 0.5),
             xytext=(15, 0),
             textcoords='offset points',
             va='center', ha='left',
             arrowprops=dict(arrowstyle='->'))
plt.annotate('', (omegac-0.5*delta_omega, 0.5),
             xytext=(-15, 0),
             textcoords='offset points',
             va='center', ha='right',
             arrowprops=dict(arrowstyle='->'))

plt.xlabel('Frequency (radians per sample)')
plt.ylabel(r'$|H(e^{j\omega})|$')
plt.title('Lowpass Filter Design Specifications', fontsize=10)
plt.tight_layout()
plt.savefig("lowpass_design_specs.pdf")
