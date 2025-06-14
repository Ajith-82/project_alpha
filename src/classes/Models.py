import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from typing import Optional

def s_model(info: dict) -> tfd.JointDistributionSequentialAutoBatched:
    """
    Define and return graphical model.

    Parameters
    ----------
    info: dict
        Data information.
    """
    tt = info['tt']
    order_scale = info['order_scale']
    order =  len(order_scale) - 1

    m = [tfd.Normal(loc=tf.zeros([info['num_stocks'], order + 1]), scale=4 * order_scale), # phi
         tfd.Normal(loc=tf.zeros([info['num_stocks'], 1]), scale=4)] # psi

    m += [lambda psi, phi: tfd.Normal(loc=tf.tensordot(phi, tt, axes=1), scale=tf.math.softplus(psi))] # y

    return tfd.JointDistributionSequentialAutoBatched(m)

def train_s(logp: np.array, info: dict, learning_rate: float = 0.01, num_steps: int = 10000, plot_loss: bool = False) -> tuple:
    """
    It performs sequential optimization over the model parameters via Adam optimizer, training at different levels to
    provide sensible initial solutions at finer levels.

    Parameters
    ----------
    logp: np.array
        Log-price at stock-level.
    info: dict
        Data information.
    learning_rate: float
        Adam's fixed learning rate.
    num_steps: int
        Adam's fixed number of iterations.
    plot_loss: bool
        Plot loss decay.

    Returns
    -------
    It returns a tuple of trained parameters.
    """
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # stock
    model = s_model(info)

    phi, psi = (tf.Variable(tf.zeros_like(model.sample()[i])) for i in range(2))
    loss = tfp.math.minimize(lambda: -model.log_prob([phi, psi, logp]), optimizer=optimizer, num_steps=num_steps)

    if plot_loss:
        fig_name = 'loss_decay.png'
        fig = plt.figure(figsize=(10, 3))
        plt.plot(loss)
        plt.legend(["loss decay"], fontsize=12, loc="upper right")
        plt.xlabel("iteration", fontsize=12)
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Loss decay plot has been saved in this directory as {}.'.format(fig_name))

    return phi, psi

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def msis_mcs_model(info: dict, level: str = "stock") -> tfd.JointDistributionSequentialAutoBatched:
    """
    Define and return graphical model.

    Parameters
    ----------
    info: dict
        Data information.
    level: str
        Level of the model; possible candidates are "stock", "industry", "sector" and "market".
    """
    tt = info['tt']
    order_scale = info['order_scale']
    order =  len(order_scale) - 1
    num_sectors = info['num_sectors']
    sec2ind_id = info['sector_industries_id']
    ind_id = info['industries_id']

    available_levels = ["market", "sector", "industry", "stock"]
    if level not in available_levels:
        raise Exception("Selected level is unknown. Please provide one of the following levels: {}.".format(available_levels))

    m = [tfd.Normal(loc=tf.zeros([1, order + 1]), scale=4 * order_scale), # phi_m
         tfd.Normal(loc=0, scale=4)] # psi_m

    if level != "market":
        m += [lambda psi_m, phi_m: tfd.Normal(loc=tf.repeat(phi_m, num_sectors, axis=0), scale=2 * order_scale), # phi_s
              lambda phi_s, psi_m: tfd.Normal(loc=psi_m, scale=2 * tf.ones([num_sectors, 1]))] # psi_s

        if level != "sector":
            sec2ind_id = info['sector_industries_id']
            m += [lambda psi_s, phi_s: tfd.Normal(loc=tf.gather(phi_s, sec2ind_id, axis=0), scale=order_scale), # phi_i
                  lambda phi_i, psi_s: tfd.Normal(loc=tf.gather(psi_s, sec2ind_id, axis=0), scale=1)] # psi_i

            if level != "industry":
                ind_id = info['industries_id']
                m += [lambda psi_i, phi_i: tfd.Normal(loc=tf.gather(phi_i, ind_id, axis=0), scale=0.5 * order_scale), # phi
                      lambda phi, psi_i: tfd.Normal(loc=tf.gather(psi_i, ind_id, axis=0), scale=0.5)]  # psi

    if level == "market":
        m += [lambda psi_m, phi_m: tfd.Normal(loc=tf.tensordot(phi_m, tt, axes=1), scale=tf.math.softplus(psi_m))] # y
    if level == "sector":
        m += [lambda psi_s, phi_s: tfd.Normal(loc=tf.tensordot(phi_s, tt, axes=1), scale=tf.math.softplus(psi_s))] # y
    if level == "industry":
        m += [lambda psi_i, phi_i: tfd.Normal(loc=tf.tensordot(phi_i, tt, axes=1), scale=tf.math.softplus(psi_i))] # y
    if level == "stock":
        if "clusters_id" not in info:
            m += [lambda psi, phi: tfd.Normal(loc=tf.tensordot(phi, tt, axes=1), scale=tf.math.softplus(psi))] # y
        else:
            clusters_id = info['clusters_id']
            num_clusters = np.max(clusters_id) + 1

            m += [tfd.Normal(loc=tf.zeros([1, order + 1]), scale=4 * order_scale),  # chi_m
                  tfd.Normal(loc=0, scale=4)]  # xi_m

            m += [lambda xi_m, chi_m: tfd.Normal(loc=tf.repeat(chi_m, num_clusters, axis=0), scale=2 * order_scale), # chi_c
                  lambda chi_c, xi_m: tfd.Normal(loc=xi_m, scale=2 * tf.ones([num_clusters, 1]))]  # xi_c

            m += [lambda xi_c, chi_c: tfd.Normal(loc=tf.gather(chi_c, clusters_id, axis=0), scale=order_scale), # chi
                  lambda chi, xi_c: tfd.Normal(loc=tf.gather(xi_c, clusters_id, axis=0), scale=1)] # xi

            m += [lambda xi, chi, xi_c, chi_c, xi_m, chi_m, psi, phi:\
                      tfd.Normal(loc=tf.tensordot(phi + chi, tt, axes=1), scale=tf.math.softplus(psi + xi))]  # y

    return tfd.JointDistributionSequentialAutoBatched(m)

def train_msis_mcs(
    logp: np.array,
    info: dict,
    learning_rate: float = 0.01,
    num_steps: int = 10000,
    plot_losses: bool = False,
    initial_params: Optional[tuple] = None,
) -> tuple:
    """
    It performs sequential optimization over the model parameters via Adam optimizer, training at different levels to
    provide sensible initial solutions at finer levels.

    Parameters
    ----------
    logp: np.array
        Log-price at stock-level.
    info: dict
        Data information.
    learning_rate: float
        Adam's fixed learning rate.
    num_steps: int
        Adam's fixed number of iterations.
    plot_losses: bool
        If True, a losses decay plot is saved in the current directory.
    initial_params: tuple | None
        Optional tuple providing initial values for the hierarchical model
        parameters. If not given, parameters are initialized at zero.

    Returns
    -------
    It returns a tuple of trained parameters.
    """
    num_steps_l = int(np.ceil(num_steps // 4))

    # Each optimization stage uses a fresh optimizer instance.  Adam does not
    # allow updating variables it wasn't originally built with, so re-create it
    # before training parameters at a new hierarchy level.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # market
    model = msis_mcs_model(info, "market")
    sample = model.sample()
    if initial_params:
        phi_m_init, psi_m_init = initial_params[0], initial_params[1]
    else:
        phi_m_init = psi_m_init = None
    phi_m = tf.Variable(
        tf.convert_to_tensor(phi_m_init, dtype=sample[0].dtype)
        if phi_m_init is not None
        else tf.zeros_like(sample[0])
    )
    psi_m = tf.Variable(
        tf.convert_to_tensor(psi_m_init, dtype=sample[1].dtype)
        if psi_m_init is not None
        else tf.zeros_like(sample[1])
    )
    loss_m = tfp.math.minimize(
        lambda: -model.log_prob([phi_m, psi_m, logp.mean(0, keepdims=1)]),
        optimizer=optimizer,
        num_steps=num_steps_l,
    )

    # sector
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = msis_mcs_model(info, "sector")
    phi_m, psi_m = tf.constant(phi_m), tf.constant(psi_m)
    sample = model.sample()
    if initial_params:
        phi_s_init, psi_s_init = initial_params[2], initial_params[3]
    else:
        phi_s_init = psi_s_init = None
    phi_s = tf.Variable(
        tf.convert_to_tensor(phi_s_init, dtype=sample[2].dtype)
        if phi_s_init is not None
        else tf.zeros_like(sample[2])
    )
    psi_s = tf.Variable(
        tf.convert_to_tensor(psi_s_init, dtype=sample[3].dtype)
        if psi_s_init is not None
        else tf.zeros_like(sample[3])
    )
    logp_s = np.array([
        logp[np.where(np.array(info['sectors_id']) == k)[0]].mean(0)
        for k in range(info['num_sectors'])
    ])
    loss_s = tfp.math.minimize(
        lambda: -model.log_prob([phi_m, psi_m, phi_s, psi_s, logp_s]),
        optimizer=optimizer,
        num_steps=num_steps_l,
    )

    # industry
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = msis_mcs_model(info, "industry")
    phi_s, psi_s = tf.constant(phi_s), tf.constant(psi_s)
    sample = model.sample()
    if initial_params:
        phi_i_init, psi_i_init = initial_params[4], initial_params[5]
    else:
        phi_i_init = psi_i_init = None
    phi_i = tf.Variable(
        tf.convert_to_tensor(phi_i_init, dtype=sample[4].dtype)
        if phi_i_init is not None
        else tf.zeros_like(sample[4])
    )
    psi_i = tf.Variable(
        tf.convert_to_tensor(psi_i_init, dtype=sample[5].dtype)
        if psi_i_init is not None
        else tf.zeros_like(sample[5])
    )
    logp_i = np.array([logp[np.where(np.array(info['industries_id']) == k)[0]].mean(0) for k in range(info['num_industries'])])
    loss_i = tfp.math.minimize(
        lambda: -model.log_prob([phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, logp_i]),
        optimizer=optimizer,
        num_steps=num_steps_l,
    )

    # stock
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = msis_mcs_model(info, "stock")
    phi_i, psi_i = tf.constant(phi_i), tf.constant(psi_i)
    sample = model.sample()
    if 'clusters_id' in info:
        if initial_params and len(initial_params) >= 14:
            phi_init, psi_init, chi_m_init, xi_m_init, chi_c_init, xi_c_init, chi_init, xi_init = initial_params[6:14]
        elif initial_params:
            phi_init, psi_init = initial_params[6:8]
            chi_m_init = xi_m_init = chi_c_init = xi_c_init = chi_init = xi_init = None
        else:
            phi_init = psi_init = chi_m_init = xi_m_init = chi_c_init = xi_c_init = chi_init = xi_init = None
        phi = tf.Variable(
            tf.convert_to_tensor(phi_init, dtype=sample[6].dtype)
            if phi_init is not None
            else tf.zeros_like(sample[6])
        )
        psi = tf.Variable(
            tf.convert_to_tensor(psi_init, dtype=sample[7].dtype)
            if psi_init is not None
            else tf.zeros_like(sample[7])
        )
        chi_m = tf.Variable(
            tf.convert_to_tensor(chi_m_init, dtype=sample[8].dtype)
            if chi_m_init is not None
            else tf.zeros_like(sample[8])
        )
        xi_m = tf.Variable(
            tf.convert_to_tensor(xi_m_init, dtype=sample[9].dtype)
            if xi_m_init is not None
            else tf.zeros_like(sample[9])
        )
        chi_c = tf.Variable(
            tf.convert_to_tensor(chi_c_init, dtype=sample[10].dtype)
            if chi_c_init is not None
            else tf.zeros_like(sample[10])
        )
        xi_c = tf.Variable(
            tf.convert_to_tensor(xi_c_init, dtype=sample[11].dtype)
            if xi_c_init is not None
            else tf.zeros_like(sample[11])
        )
        chi = tf.Variable(
            tf.convert_to_tensor(chi_init, dtype=sample[12].dtype)
            if chi_init is not None
            else tf.zeros_like(sample[12])
        )
        xi = tf.Variable(
            tf.convert_to_tensor(xi_init, dtype=sample[13].dtype)
            if xi_init is not None
            else tf.zeros_like(sample[13])
        )

        loss = tfp.math.minimize(
            lambda: -model.log_prob([phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi, chi_m, xi_m, chi_c, xi_c, chi, xi, logp]),
                                 optimizer=optimizer, num_steps=num_steps_l)
    else:
        if initial_params:
            phi_init, psi_init = initial_params[6], initial_params[7]
        else:
            phi_init = psi_init = None
        phi = tf.Variable(
            tf.convert_to_tensor(phi_init, dtype=sample[6].dtype)
            if phi_init is not None
            else tf.zeros_like(sample[6])
        )
        psi = tf.Variable(
            tf.convert_to_tensor(psi_init, dtype=sample[7].dtype)
            if psi_init is not None
            else tf.zeros_like(sample[7])
        )
        loss = tfp.math.minimize(lambda: -model.log_prob([phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi, logp]),
                                 optimizer=optimizer, num_steps=num_steps_l)

    if plot_losses:
        fig_name = 'losses_decay.png'
        fig = plt.figure(figsize=(20, 3))
        plt.subplot(141)
        plt.title("market-level", fontsize=12)
        plt.plot(loss_m)
        plt.subplot(142)
        plt.title("sector-level", fontsize=12)
        plt.plot(loss_s)
        plt.subplot(143)
        plt.title("industry-level", fontsize=12)
        plt.plot(loss_i)
        plt.subplot(144)
        plt.title("stock-level", fontsize=12)
        plt.plot(loss)
        plt.legend(["loss decay"], fontsize=12, loc="upper right")
        plt.xlabel("iteration", fontsize=12)
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Losses decay plot has been saved in this directory as {}.'.format(fig_name))

    params = (phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi)
    if 'clusters_id' in info:
        params += (chi_m, xi_m, chi_c, xi_c, chi, xi)
    return params

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def conj_lin_model(logp: np.array):
    """
    A Gaussian-Gaussian conjugate model that sequentially predicts the next log-price.

    Parameters
    ----------
    logp: np.array
        Log-prices at stock-level.

    Returns
    -------
    pred_all: np.array
        Prediction for each stock of the last T-2 prices and of the next price, where T is the number of dates.
    """
    num_stocks, T = logp.shape

    mu0 = np.zeros((num_stocks, 2, 1))
    cov0_inv = np.eye(2)[None].repeat(num_stocks, 0)

    gamma2 = np.ones((num_stocks, 1))

    ones = np.ones((num_stocks, 1))

    pred_all = np.zeros((num_stocks, T - 1))

    for t in range(T-1):
        y = logp[:, t, None]
        ytilde = np.hstack((ones, y))

        cov_inv = cov0_inv + ytilde[:, :, None] * ytilde[:, None] / gamma2[:, None]
        det = cov_inv[:, 0, 0] * cov_inv[:, 1, 1] - cov_inv[:, 0, 1] ** 2
        cov = np.zeros((num_stocks, 2, 2))
        cov[:, 0, 0] = cov_inv[:, 1, 1] / det
        cov[:, 1, 1] = cov_inv[:, 0, 0] / det
        cov[:, 0, 1] = -cov_inv[:, 0, 1] / det
        cov[:, 1, 0] = -cov_inv[:, 1, 0] / det

        mu = np.sum(cov0_inv * mu0, 1) + ytilde * logp[:, t + 1, None] / gamma2
        mu = np.sum(cov * mu[:, None, :], 2, keepdims=True)

        mu0 = mu
        cov0_inv = cov_inv

        gamma2 = np.var(logp[:, :t + 2], 1, keepdims=True) + 1e-6

        pred_all[:, t] = np.sum(mu.squeeze() * np.hstack((ones, logp[:, t + 1, None])), 1)

    return pred_all
