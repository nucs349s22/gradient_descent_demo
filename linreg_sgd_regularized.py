import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=18)

np.random.seed(42)

def hypothesis(x, theta1):
    """Our "hypothesis function", a straight line through the origin."""
    return theta1*x

def cost_func(theta1, y, x):
    """The cost function, J(theta1) describing the goodness of fit."""
    theta1 = np.atleast_2d(np.asarray(theta1))
    return np.average((y-hypothesis(x, theta1))**2, axis=1)/2

def regularization_func(theta1, p):
    if p == 2:
        return np.square(theta1)
    elif p ==1:
        return np.abs(theta1)
    else:
        raise ValueError(f"Unk p {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=int, default=2)
    parser.add_argument("--lambda_", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_steps", type=int, default=5)
    args = parser.parse_args()

    # The data to fit
    m = 20
    # sigma = 0.5
    sigma = 0.1
    theta1_true = 0.5
    x = np.linspace(-1,1,m)
    y = theta1_true * x + np.random.normal(0, sigma, m)

    ax1_xlim = [-0.5, 1]

    # The plot: LHS is the data, RHS will be the cost function.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
    fig.suptitle(f"$L_{args.lp}$, $\\lambda$={args.lambda_}, lr={args.lr}")
    ax[0].scatter(x, y, marker='x', s=40, color='k')
    truth = ax[0].plot(x, hypothesis(x, theta1_true), alpha=0.5,
               lw=5, color='black', label=r'$\theta_1 = {:.3f}$'.format(theta1_true))

    # First construct a grid of theta1 parameter pairs and their corresponding
    # cost function values.
    theta1_grid = np.linspace(*ax1_xlim, 100)
    J_grid = cost_func(theta1_grid[:,np.newaxis], y, x)
    reg_grid = regularization_func(theta1_grid, args.lp)
    combo_grid = J_grid + args.lambda_ * reg_grid

    # The cost function as a function of its single parameter, theta1.
    ax[1].plot(theta1_grid, J_grid, color='blue', alpha=0.5, lw=1, label=r"$L_D(\theta, X, y)$", zorder=0.5)
    ax[1].plot(theta1_grid, reg_grid, color='red', alpha=0.5, lw=1, label=r"$L_R(\theta)$", zorder=0.5)
    ax[1].plot(theta1_grid, combo_grid, color='purple', alpha=0.9, lw=2, label=r"$L_{D+R}$", zorder=0.5)

    # Take num_steps steps with learning rate lr down the steepest gradient,
    # starting at theta1 = 0.
    theta1 = [-0.5]
    J = [cost_func(theta1[0], y, x)[0] + args.lambda_ * regularization_func(theta1[0], args.lp)]
    for j in range(args.num_steps - 1):
        last_theta1 = theta1[-1]
        this_theta1 = last_theta1 - args.lr / m * np.sum(
                                        (hypothesis(x, last_theta1) - y) * x)
        if args.lambda_ > 0:
            tmp_theta = this_theta1
            if args.lp == 2:
                this_theta1 -= args.lr * 2 * args.lambda_ * last_theta1
            elif args.lp == 1:
                this_theta1 -= args.lr * args.lambda_ * np.sign(last_theta1)
            # print(f"{tmp_theta} -> {this_theta1}")

        theta1.append(this_theta1)
        J.append(cost_func(this_theta1, y, x)[0] + args.lambda_ * regularization_func(this_theta1, args.lp))

    buffer = 0.01
    colors = ['g', 'y', 'aqua', 'c', 'DeepPink', 'orange'][:args.num_steps]
    # ax[1].scatter(theta1, J, c=colors, s=10, lw=0)
    ax[1].set_xlim(*ax1_xlim)

    ymin = np.min(np.stack([combo_grid, J_grid, reg_grid], axis=1)) - buffer
    ymax = np.max(np.stack([combo_grid, J_grid, reg_grid], axis=1)) + buffer
    ax[1].set_ylim(ymin, ymax)
    ax[0].set_ylim(-0.5,0.5)
    ax[1].set_xlabel(r'$\theta_1$')
    ax[1].set_ylabel(r'$J(\theta_1)$')
    ax[1].set_title('Loss function')
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_title('Hypotheses')

    handles = truth
    ax[0].legend(handles=handles, loc='upper left', fontsize='x-small')
    ax[1].legend(loc='upper left', fontsize='x-small')

    plt.tight_layout()
    # plt.show(block=False)

    count = 0
    dir_ = f"lmbda{args.lambda_}_lp{args.lp}_lr{args.lr}"
    if not os.path.isdir(dir_):
        os.mkdir(dir_)
    plt.savefig(f"{dir_}/{count}.png")

    # Annotate the cost function plot with coloured points indicating the
    # parameters chosen and red arrows indicating the steps down the gradient.
    # Also plot the fit function on the LHS data plot in a matching colour.
    hypo = ax[0].plot(x, hypothesis(x, theta1[0]), color=colors[0], lw=2,
                      label=r'$\theta_1 = {:.3f}$'.format(theta1[0]))
    ax[1].scatter(theta1[0], J[0], c=colors[0], s=40, lw=0, zorder=1)
    handles.extend(hypo)
    ax[0].legend(handles=handles, loc='upper left', fontsize='x-small')
    # input()
    # plt.pause(0.1)  
    count += 1
    plt.savefig(f"{dir_}/{count}.png")
    for j in range(1, args.num_steps):
        ax[1].annotate('', xy=(theta1[j], J[j]), xytext=(theta1[j-1], J[j-1]),
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')
        ax[1].scatter(theta1[j], J[j], c=colors[j], s=64, zorder=1)
        hypo = ax[0].plot(x, hypothesis(x, theta1[j]), color=colors[j], lw=2,
                             label=r'$\theta_1 = {:.3f}$'.format(theta1[j]))
        handles.extend(hypo)
        ax[0].legend(handles=handles, loc='upper left', fontsize='x-small')
        # input()
        count += 1
        plt.savefig(f"{dir_}/{count}.png")
        # plt.pause(0.1)  

    # Labels, titles and a legend.
    # kplt.show()


if __name__ == "__main__":
    main()
