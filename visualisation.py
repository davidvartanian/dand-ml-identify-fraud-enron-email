import matplotlib.pyplot as plt


def scatter_plot(X, Y, xlabel='X', ylabel='Y', xplot=None, yplot=None):
    """
    Render a scatter plot
    """
    for x_point, y_point in zip(X, Y):
        plt.scatter(x_point[0:1], y_point)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xplot is not None and yplot is not None:
        plt.plot(xplot, yplot, color="b") 
    plt.show()