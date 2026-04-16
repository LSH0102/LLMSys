import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    # Example placeholder values - replace with actual benchmark results
    # Data Parallel comparison
    single_mean, single_std = 10.0, 0.5  # Single GPU training time
    device0_mean, device0_std = 6.0, 0.3  # GPU0 training time with 2 GPUs
    device1_mean, device1_std = 6.0, 0.3  # GPU1 training time with 2 GPUs
    
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'ddp_vs_rn.png')

    # Pipeline Parallel comparison
    pp_mean, pp_std = 8.0, 0.4  # Pipeline parallel training time
    mp_mean, mp_std = 12.0, 0.6  # Model parallel training time
    
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'pp_vs_mp.png')
    
    print("Plots saved: ddp_vs_rn.png, pp_vs_mp.png")