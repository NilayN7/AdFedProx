import argparse
from data import get_MNIST, plot_samples
from model import CNN
from fedprox import FedProx
import matplotlib.pyplot as plt


def plot_acc_loss(title: str, loss_hist: list, acc_hist: list):
    plt.figure()

    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    lines = plt.plot(loss_hist)
    plt.title("Loss")
    plt.legend(lines, ["C1", "C2", "C3"])

    plt.subplot(1, 2, 2)
    lines = plt.plot(acc_hist)
    plt.title("Accuracy")
    plt.legend(lines, ["C1", "C2", "C3"])
    plt.savefig(title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="basic")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--setting", type=str, default="iid")
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--project", type=bool, default=True)
    parser.add_argument("--q", type=float, default=0.2)
    args = parser.parse_args()

    # TODO: Add other models: basic ResNetand other baselines (VGG, AlexNet, ...)
    if args.model == "basic":
        model_0 = CNN()

    # TODO: Add other datasets: CIFAR10, CIFAR100, CelebA (probably)
    if args.dataset == "mnist" and args.setting == "iid":
        mnist_train_dls, mnist_test_dls = get_MNIST(
            "iid", n_samples_train=200, n_samples_test=100,
            n_clients=args.n_clients, batch_size=args.n_clients, shuffle=True
        )

    elif args.dataset == "mnist" and args.setting == "non-iid":
        mnist_train_dls, mnist_test_dls = get_MNIST(
            "non_iid", n_samples_train=200, n_samples_test=100,
            n_clients=args.n_clients, batch_size=args.batch_size, shuffle=True
        )

    if args.visualize:
        plot_samples(next(iter(mnist_train_dls[0])), 0, "Client 1")
        plot_samples(next(iter(mnist_train_dls[1])), 0, "Client 2")
        plot_samples(next(iter(mnist_train_dls[2])), 0, "Client 3")

    # FedAvg
    model_f, loss_hist_FA, acc_hist_FA = FedProx(
        model_0, mnist_train_dls, args.n_iter,
        mnist_test_dls, epochs=args.epochs, lr=args.lr, mu=0, project=args.project, q=args.q
    )

    # FedProx
    model_f, loss_hist_FP, acc_hist_FP = FedProx(
        model_0, mnist_train_dls, args.n_iter,
        mnist_test_dls, epochs=args.epochs, lr=args.lr, mu=0.3, project=args.project, q=args.q)

    # comparison
    plot_acc_loss("ProjFedAvg MNIST-iid", loss_hist_FA, acc_hist_FA)

