# Adapted from Christian Hill's code at:
# https://scipython.com/blog/visualizing-the-gradient-descent-method/

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
plt.rc('font', size=18)


def main(**kwargs):
  # theta_true = np.array([0, 2.5, -4.3, 8.1,])
  # theta_true = np.array([-2, 1.5, 0] + [0] * 3)
  # fit_dim = 2
  # dims_to_plot = (0, 1)
  # N = 16
  # alpha = 0.5
  # max_to_plot = 4

  theta_true = kwargs['theta_true']
  fit_dim = kwargs['fit_dim']
  dims_to_plot1 = kwargs['dims_to_plot1']
  dims_to_plot2 = kwargs['dims_to_plot2']
  N = kwargs['N']
  alpha = kwargs['alpha']
  max_to_plot = kwargs['max_to_plot']

  m = 8
  sigma_1 = 3.6
  sigma_2 = 5
  sigma_2_prob = 0.0

  d = np.shape(theta_true)[0]
  x = np.linspace(-1,1,m)
  x_vec = np.transpose(np.array([x ** i for i in range(d)]))
  y = np.matmul(x_vec, theta_true)
  err_choice = np.random.binomial(1, sigma_2_prob, m)
  y += (1 - err_choice) * np.random.normal(0, sigma_1, m)
  y += err_choice * np.random.normal(0, sigma_2, m)

  # The plot: LHS is the data, RHS will be the cost function.
  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14,7))
  ax[0].scatter(x, y, marker='x', s=40, color='k')

  def cost_func(theta, l1=0, l2=0):
      """The cost function, J(theta0, theta1) describing the goodness of fit."""
      # theta = np.atleast_3d(np.asarray(theta))
      # theta1 = np.atleast_3d(np.asarray(theta1))
      cost = (y-hypothesis(x, theta))**2
      # print(cost.shape)
      return np.average(np.square(cost))

  def hypothesis(x, theta):
      """Our "hypothesis function", a straight line."""
      x_vec = np.transpose(np.array([x ** i for i in range(d)]))
      # print(x_vec.shape, theta.shape)
      return np.matmul(x_vec, theta)

  # First construct a grid of (theta0, theta1) parameter pairs and their
  # corresponding cost function values.

  fit_dim = min(fit_dim, d)
  theta = [np.zeros(d)]
  for i, dims_to_plot in enumerate([dims_to_plot1, dims_to_plot2]):
    for j, dim in enumerate(dims_to_plot):
      theta[0][dim] = 3 * (-1) ** (j)
    assert len(dims_to_plot) == 2
    grid_points = 101
    theta0_grid = np.linspace(-4, 5, grid_points)
    theta1_grid = np.linspace(-4, 6, grid_points)
    X, Y = np.meshgrid(theta0_grid, theta1_grid)
    theta0_grid = np.repeat(
        np.reshape(theta0_grid, (grid_points, 1))[np.newaxis, :, :], grid_points, axis=0)
    theta1_grid = np.repeat(
        np.reshape(theta1_grid, (grid_points, 1))[:, np.newaxis, :], grid_points, axis=1)
    theta_grid = np.concatenate([theta0_grid, theta1_grid], axis=2)

    #theta_list = theta_grid.reshape(-1, 2).tolist()
    theta_list = []
    for param_tuple in theta_grid.reshape(-1, 2).tolist():
      tmp = theta_true.copy()
      tmp[dims_to_plot[0]] = param_tuple[0]
      tmp[dims_to_plot[1]] = param_tuple[1]
      theta_list.append(tmp)

    # theta_zeros = [theta_true for _ in range(m)]
      # theta_list = [[0] * start + theta_item + [0] * (d - start - 2)
      #               for theta_item in theta_list]
    J_list = [cost_func(np.array(theta_item)) for theta_item in theta_list]
    J_grid = np.array(J_list).reshape(grid_points, grid_points)
                       

    # A labeled contour plot for the RHS cost function
    contours = ax[i + 1].contour(X, Y, J_grid, [200, 400, 600, 800, 1600, 3200, 6400])
    ax[i + 1].clabel(contours, fmt='%1d')
    # The target parameter values indicated on the cost function contour plot
    ax[i + 1].scatter([theta_true[dims_to_plot[0]]]*2,[theta_true[dims_to_plot[1]]]*2,
                  s=[50,10], color=['k','w'])


  # theta = [np.array((-3,-3))]
  J = [cost_func(theta[0])]
  for j in range(N-1):
      last_theta = theta[-1]
      this_theta = np.zeros(d)
      for i in range(fit_dim):
        this_theta[i] = last_theta[i] - alpha / m * np.sum(
                                        (hypothesis(x, last_theta) - y) * x ** (i))
      # this_theta[0 + start] = last_theta[0 + start] - alpha / m * np.sum(
      #                                 (hypothesis(x, last_theta) - y) * x ** (0 + start))
      # this_theta[1 + start] = last_theta[1 + start] - alpha / m * np.sum(
      #                                 (hypothesis(x, last_theta) - y) * x ** (1 + start))
      theta.append(this_theta)
      J.append(cost_func(this_theta))


  # Annotate the cost function plot with coloured points indicating the
  # parameters chosen and red arrows indicating the steps down the gradient.
  # Also plot the fit function on the LHS data plot in a matching colour.
  colors = ['b', 'g', 'm', 'c', 'orange']
  while len(colors) < min(N, 1 + max_to_plot):
    colors += colors
  colors = colors[:min(N, 1 + max_to_plot)]


  def to_func(theta, end=np.inf):
    retval = "f(x) = {:.2f}".format(theta[0])
    end = min(end, len(theta))
    for i in range(1, end):
      sign = "-" if theta[i] < 0 else "+"
      retval += " {} {:.2f}x".format(sign, abs(theta[i]))
      if i > 1:
        retval += "^{}".format(i)
    return retval

  truth = ax[0].plot(x, hypothesis(x, theta_true), alpha=0.5,
             lw=5, color='black', label=r'truth: ${}$'.format(to_func(theta_true)))
  handles = truth
  
  ax[1].set_xlabel(r'$\theta_{}$'.format(dims_to_plot1[0]))
  ax[1].set_ylabel(r'$\theta_{}$'.format(dims_to_plot1[1]), labelpad=-10)
  ax[2].set_xlabel(r'$\theta_{}$'.format(dims_to_plot2[0]))
  ax[2].set_ylabel(r'$\theta_{}$'.format(dims_to_plot2[1]), labelpad=-10)
  ax[1].set_title('Loss function')
  ax[2].set_title('Loss function')
  ax[0].set_xlabel(r'$x$')
  ax[0].set_ylabel(r'$y$', labelpad=-10)
  ax[0].set_ylim(-5,5)
  ax[0].set_title('Data and fit')
  ax[0].legend(loc='upper center', bbox_to_anchor=(1.6, -0.25),
            fancybox=True, fontsize='small', handles=handles)
  plt.subplots_adjust(bottom=0.4)
  plt.margins(0,0)
  # plt.gca().xaxis.set_major_locator(plt.NullLocator())
  # plt.gca().yaxis.set_major_locator(plt.NullLocator())
  count = 0
  plt.savefig("4d_{}.png".format(count), bbox_inches='tight', pad_inches=0)

  hypo = ax[0].plot(x, hypothesis(x, theta[0]), color=colors[0], lw=2,
             label=r'${}$'.format(to_func(theta[0], end=fit_dim), end=fit_dim))
  handles.extend(hypo)
  ax[0].legend(loc='upper center', bbox_to_anchor=(1.6, -0.25),
            fancybox=True, fontsize='small', handles=handles)

  color_list = [colors[0]]
  xy_list = {}
  for i, dims_to_plot in enumerate([dims_to_plot1, dims_to_plot2]):
    xy_list[i] = [[theta[0][i] for i in dims_to_plot]]
    ax[i + 1].scatter(*zip(*xy_list[i]), c=color_list, s=40, lw=0)
  count += 1
  plt.savefig("4d_{}.png".format(count), bbox_inches='tight', pad_inches=0)
  for j in range(1,N):
      if N <= max_to_plot or j % (N // max_to_plot) == 0:
        color_j = len(color_list)
        hypo = ax[0].plot(x, hypothesis(x, theta[j]), color=colors[color_j], lw=2,
               label=r'${}$'.format(to_func(theta[j], end=fit_dim)))
        handles.extend(hypo)
        ax[0].legend(loc='upper center', bbox_to_anchor=(1.6, -0.25),
                  fancybox=True, fontsize='small', handles=handles)

        color_list.append(colors[color_j])
        for i, dims_to_plot in enumerate([dims_to_plot1, dims_to_plot2]):
          tup = [theta[j][i] for i in dims_to_plot]
          ax[i + 1].annotate('', xy=tup, xytext=xy_list[i][-1],
                         arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                         va='center', ha='center')
          xy_list[i].append(tup)
          ax[i + 1].scatter(*zip(*xy_list[i]), c=color_list, s=40, lw=0)
        count += 1
        plt.savefig("4d_{}.png".format(count), bbox_inches='tight', pad_inches=0)


  # Labels, titles and a legend.
  # axbox = ax[0].get_position()
  # Position the legend by hand so that it doesn't cover up any of the lines.
  # ax[0].legend(loc=(axbox.x0+0.1*axbox.width, axbox.y0+0.1*axbox.height),
  #              fontsize='small')
  # ax[0].legend(fontsize='x-small', )
  # ax[0].legend(fontsize='x-small', bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

  # ax[0].legend(loc=7, fontsize='x-small',)
  # plt.tight_layout(pad=1.0)



  # plt.show()


def good_fit():
  main(
    theta_true = np.array([0, 2.5, 0, 0]),
    fit_dim = 4,
    dims_to_plot1 = (0, 1),
    dims_to_plot2 = (2, 3),
    N = 32,
    alpha = 0.3,
    max_to_plot = 4,
  )


if __name__ == "__main__":
  good_fit()
