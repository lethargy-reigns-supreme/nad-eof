import numpy as np
import matplotlib.pyplot as plt

def open_data(path):
    corrs =[]
    file = open(path, 'r')
    for line in file:
        data = line.strip()
        s = slice(-4, len(data))
        corrs.append(float(data[s]))
    file.close()
    return corrs

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12})

models = ['INM-CM4-8', 'NorESM2-MM', 'MIROC6', 'TaiESM1', 'ACCESS-CM2', 'INM-CM5-0']
corrs_hist = []
corrs_245_JJA = []
corrs_245_MAM = []
corrs_370_JJA = []
corrs_370_MAM = []
corrs_585_JJA = []
corrs_585_MAM = []

w = .3

x1 = np.arange(len(models))
x2 = [i+w for i in x1]
x_labels = [i+w/2 for i in x1]

corrs_hist = open_data('bai-data-hist-6m.txt')
corrs_245_JJA = open_data('data-245-JJA.txt')
corrs_245_MAM = open_data('data-245-MAM.txt')
corrs_370_JJA = open_data('data-370-JJA.txt')
corrs_370_MAM = open_data('data-370-MAM.txt')
# corrs_585_JJA = open_data('bai-data-JJA-6m.txt')
# corrs_585_MAM = open_data('bai-data-MAM-6m.txt')


fig, ax = plt.subplots(3, 1, figsize=(8, 11))

#historical bars
ax[0].set_title('Historical')
ax[0].set_ylim(top=1)
bars_hist = ax[0].bar(x1, corrs_hist, width=w, label='Summer')
# ax[0].set_title('Changes in correlation between observed and predicted NAD based on emissions')
ax[0].bar_label(bars_hist)
ax[0].set_xticks(x1, models)
ax[0].set_ylabel("Pearson's Correlation")
ax[0].legend(loc='upper right')

#ssp245 bars
ax[1].set_title('SSP245')

ax[1].set_ylim(top=1)
bars_245_MAM = ax[1].bar(x1, corrs_245_MAM, width=w, label='Spring', color='m')
bars_245_JJA = ax[1].bar(x2, corrs_245_JJA, width=w, label='Summer', color='g')
ax[1].bar_label(bars_245_MAM)
ax[1].bar_label(bars_245_JJA)
ax[1].set_xticks(x_labels, models)
ax[1].set_ylabel("Pearson's Correlation")

ax[1].legend(loc='upper left')

#ssp370 bars
ax[2].set_title('SSP370')

ax[2].set_ylim(top=1)
bars_370_MAM = ax[2].bar(x1, corrs_370_MAM, width=w, label='Spring', color='m')
bars_370_JJA = ax[2].bar(x2, corrs_370_JJA, width=w, label='Summer', color='g')
ax[2].bar_label(bars_370_MAM)
ax[2].bar_label(bars_370_JJA)
ax[2].set_xticks(x_labels, models)
ax[2].set_xlabel("Models")
ax[2].set_ylabel("Pearson's Correlation")
ax[2].legend(loc='upper left')

# ax[3].set_ylim(top=1)
# bars_585_MAM = ax[3].bar(x1, corrs_585_MAM, width=w, label='MAM SSP585', color='m')
# bars_585_JJA = ax[3].bar(x2, corrs_585_JJA, width=w, label='JJA SSP585', color='g')
# ax[3].bar_label(bars_585_MAM)
# ax[3].bar_label(bars_585_JJA)
# ax[3].set_xticks(x_labels, models)
# ax[3].legend(loc='upper left')

plt.tight_layout()
# plt.savefig('full-barchart-bai.png')
plt.show()
