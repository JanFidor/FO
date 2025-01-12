from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_image_predictions(states, n_patterns, n_transitions, prefix):
    if (n_patterns == 1):
        fig, ax = plt.subplots(1, n_transitions, figsize=(10, 5))
        for i in range(n_transitions):
            ax[i].matshow(states[0][i].reshape((28, 28)), cmap='gray')  
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            title = 'train_0' if i == 0 else 'test_0' if i == 1 else 'predict_0'
            ax[i].set_title(title)
    else:
        fig, ax = plt.subplots(n_patterns, n_transitions, figsize=(10, 5))
        for j in range(n_patterns):
            for k in range(n_transitions):
                ax[j, k].matshow(states[j][k].reshape((28, 28)), cmap='gray')
                ax[j, k].set_xticks([])
                ax[j, k].set_yticks([])
                pfx = 'train' if k == 0 else 'test' if k == 1 else 'predict'
                sfx = f'_{j}'
                ax[j, k].set_title(pfx + sfx)

    save_path = Path('./png') / f'{prefix}_mnist_image_prediction'
    plt.savefig(save_path)
    plt.show()
    plt.close()


def draw_energy_transition(energies, prefix):
    x_axis = np.arange(len(energies[0]))
    for energy_idx in range(len(energies)):
        plt.plot(x_axis, energies[energy_idx], label=f'pattern_{energy_idx}')
    plt.legend()
    save_path = Path('./png') / f'{prefix}_energy_transition'
    plt.savefig(save_path)
    plt.close()


def draw_weights_history(data_frames, prefix):
    num_frames = len(data_frames)

    def update_heatmap(frame, ax):
        ax.clear()
        sns.heatmap(data_frames[frame], ax=ax, cmap="coolwarm", cbar=False, annot=False, xticklabels=False, yticklabels=False)
        ax.set_title(f"Weights matrix ({frame+1}/{num_frames})")

    fig, ax = plt.subplots()
    sns.heatmap(data_frames[0], ax=ax, cmap="coolwarm", cbar=False, annot=False, xticklabels=False, yticklabels=False)
    ax.set_title(f"Weights matrix (1/{num_frames})")
    anim = FuncAnimation(fig, update_heatmap, frames=num_frames, fargs=(ax,), interval=500)
    save_path = Path('./png') / f'{prefix}_weights_history.gif'
    anim.save(save_path, writer='pillow')
    plt.close()
