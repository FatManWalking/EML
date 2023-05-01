# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch

# Convinience functions
def plot_model(model=None, x_train=None, y_train=None, x_test=None, y_test=None):
    # Visualize data
    plt.plot(
        torch.linspace(0, 1, 1000),
        ground_truth_function(torch.linspace(0, 1, 1000)),
        label="Ground truth",
    )
    plt.plot(x_train, y_train, "ob", label="Train data")
    plt.plot(x_test, y_test, "xr", label="Test data")
    # Visualize model
    if model is not None:
        plt.plot(
            torch.linspace(0, 1, 1000),
            model(torch.linspace(0, 1, 1000)),
            label=f"Model of degree: {model.degree()}",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.show()


# Generate data
n_samples = 11
noise_amplitude = 0.15


def ground_truth_function(x):
    # Generate data of the form sin(2 * Pi * x)
    # ---- Fill in the following:

    result = np.sin(2 * np.pi * x)

    return result


torch.manual_seed(42)

x_test = torch.linspace(0, 1, n_samples)
y_test = ground_truth_function(x_test) + torch.normal(
    0.0, noise_amplitude, size=(n_samples,)
)
x_train = torch.linspace(0, 1, n_samples)
y_train = ground_truth_function(x_train) + torch.normal(
    0.0, noise_amplitude, size=(n_samples,)
)

# Test plotting
plot_model(model=None, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
plt.savefig("Initial_data.png")
plt.clf()


# Model fitting


def error_function(model, x_data, y_data):
    y_pred = model(x_data)

    # ---- Fill with the error function from the lecture
    # Regularization Error(w) = 0.5 * sum( (h(xn, w) - tn)^2 ) + 0.5 * lambda * ||w||
    # Since it is NON_regularized lambda = 0 and the last part of the term could be omitted
    lambda_ = 0.0

    error = 0.5 * torch.sum(torch.square(y_pred - y_data)) + 0.5 * np.linalg.norm(
        model.coef
    )
    return error


def rms_error_function(model, x_data, y_data, lambda_):
    y_pred = model(x_data)

    rmse_loss = torch.sqrt(torch.mean(torch.square(y_pred - y_data)))
    error = rmse_loss + 0.5 * lambda_ * np.linalg.norm(model.coef)

    return error


def inital():
    model_degree = 3

    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
    train_err = error_function(model, x_train, y_train)
    test_err = error_function(model, x_test, y_test)

    print(f"{train_err=}, {test_err=}")

    # Result plotting
    plot_model(model, x_train, y_train, x_test, y_test)
    plt.savefig("Initial_fit.png")
    plt.clf()


# ---- Continue with the exercises on the degree of the polynomial and the exploration of data size


def degree():
    for model_degree in range(0, 12):

        print(f"Fitting degree {model_degree}")

        model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
        train_err = rms_error_function(model, x_train, y_train, 0)
        test_err = rms_error_function(model, x_test, y_test, 0)

        print(f"{train_err=}, {test_err=}")

        # Result plotting
        plot_model(model, x_train, y_train, x_test, y_test)
        plt.savefig(f"Poly_fit_{model_degree}.png")
        plt.clf()


def data_size(n_samples=10):

    treshold = 0.001
    noise_amplitude = 0.15

    while True:
        # Resize sample size
        n_samples = n_samples + 50

        print(f"Sample Size {n_samples}")

        # Sample the data
        x_test = torch.linspace(0, 1, n_samples)
        y_test = ground_truth_function(x_test) + torch.normal(
            0.0, noise_amplitude, size=(n_samples,)
        )
        x_train = torch.linspace(0, 1, n_samples)
        y_train = ground_truth_function(x_train) + torch.normal(
            0.0, noise_amplitude, size=(n_samples,)
        )

        # Fit the model

        model = np.polynomial.Polynomial.fit(x_train, y_train, deg=10)
        train_err = rms_error_function(model, x_train, y_train, 0)
        test_err = rms_error_function(model, x_test, y_test, 0)

        # See if both are equal
        print(f"{train_err=}, {test_err=}")
        if np.abs(train_err - test_err) < treshold:
            print(f"Equality of {train_err - test_err} achived")
            break

    # Result plotting
    plot_model(model, x_train, y_train, x_test, y_test)
    plt.savefig(f"Sample_fit_{n_samples}.png")
    plt.clf()


inital()
degree()
data_size()  # Step size of 50 starting at 10 resulted in 960. One could do a narrowed down search in that area for a more precise result
