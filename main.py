import mnist
from hopfield import HopfieldNet
from utils import draw_image_predictions, draw_energy_transition, draw_weights_history


def save_mnist_prediction(fetch: mnist.MnistForHopfield, bias: float, sync: bool) -> None:
    train = fetch.original
    test = fetch.noised
    model = HopfieldNet(inputs=train)
    states = []
    energies = []
    for data_idx in range(len(train)):
        r = model.predict(train[data_idx], bias=bias, sync=sync)
        states.append([train[data_idx], test[data_idx], r.states[-1]])
        energies.append(r.energies)

    prefix = 'sync' if sync else 'async'
    draw_image_predictions(states, len(states), len(states[0]), prefix)
    draw_energy_transition(energies, prefix)
    draw_weights_history(model.W_history, prefix)


def main():
    fetch = mnist.fetch_minist_for_hopfield(size=5, error_rate=0.14)    
    bias = 60
    save_mnist_prediction(fetch=fetch, bias=bias, sync=False)
    save_mnist_prediction(fetch=fetch, bias=bias, sync=True)


if __name__ == "__main__":
    main()
