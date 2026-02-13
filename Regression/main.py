import numpy as np

# Returns the projection of vector v onto base b
def proj(b, v):
    return (np.dot(v, b) / np.dot(b, b)) * b

def main():
    # data points
    x = np.array([0.011, 0.045, 0.117, 0.161, 0.203])
    y = np.array([0.20, 0.40, 1.20, 1.60, 2.00])
    ones = np.ones_like(x)

    # compute orthogonal basis x_hat of span(x, 1), with the other orthogonal basis vector 1
    x_hat = x - proj(ones, x)

    # projection of y onto span(x, 1) is given by the sum of projections onto orthogonal basis x_hat and 1
    # it is of form m(x_hat) + proj(1, y) = m(x - proj(1, x)) + proj(1, y)
    proj_x_hat_y = proj(x_hat, y)
    proj_1_y = proj(ones, y)

    # proj(x_hat, y) gives m(x_hat) = m(x - proj(1, x)), so m is the slope of the regression line.
    # scalar factor before elementwise multiplying with x_hat (to prevent /0 error)
    m = np.dot(y, x_hat) / np.dot(x_hat, x_hat)

    # to find the intercept, we must sum the constants -m(proj(1, x)) + proj(1, y).
    b = (proj_1_y - m * proj(ones, x))[0]

    print(f"Regression line: y = {m}x + {b}")

if __name__ == "__main__":
    main()