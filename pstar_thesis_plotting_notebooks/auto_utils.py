import numpy as np
from scipy import signal
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
from astropy.io import fits


def parse_metadata(meta_path):
    params = {}
    with fits.open(meta_path) as meta:
        meta_tbl = meta[1].data
        antenna_inds = meta_tbl["Antenna"][1::2]
        cable_flavs = np.asarray(meta_tbl["Flavors"][1::2]).astype(np.str_)
        dig_gains = meta_tbl["Gains"][1::2, :].astype(np.float64)
        antenna_numbers = meta_tbl["Tile"][1::2]
        recs = meta_tbl["Rx"][1::2]
        antenna_names = meta_tbl["TileName"][1::2].astype(np.str_)
    reordered_inds = antenna_inds.argsort()
    params["ant_nums"] = np.array(antenna_numbers[reordered_inds])
    params["cable_flavors"] = np.array(cable_flavs[reordered_inds])
    params["gains"] = np.array(dig_gains[reordered_inds])
    params["cable_delays"] = np.empty(params["cable_flavors"].shape)
    params["recs"] = np.array(recs[reordered_inds])
    params["ant_names"] = np.array(antenna_names[reordered_inds])
    for i in range(len(params["cable_flavors"])):
        j = params["cable_flavors"][i].split("_")
        if len(j) == 1:
            params["cable_delays"][i] = np.nan
        elif j[0][0] == "R":
            params["cable_delays"][i] = 2 * int(j[-1]) / (0.81 * 299.99)
        elif j[0][0] == "L":
            params["cable_delays"][i] = 2 * int(j[-1]) / (0.85 * 299.99)

    return params


def flag_fine_chans(autos, num_coarse, fine_chan_nums):
    """
    Flag a set of fine channels.
    
    Parameters
    ----------
    autos: numpy array of type float
        Array of autocorrelations with shape (Nants, Ntimes, Nfreqs, Npols)
        or shape (Nblts, Nfreqs, Npols)
    Returns
    -------
    masked_autos: numpy array of type float
        Masked array of autos with fine chan flags applied
    """

    coarse_chan_width = int(autos.shape[-2] / num_coarse)

    if np.max(fine_chan_nums) >= coarse_chan_width:
        raise ValueError("cannot flag fine channel numbers that do not exist")

    mask = np.full(coarse_chan_width, False)
    mask[fine_chan_nums] = True

    mask = np.tile(mask, num_coarse)

    masked_autos = np.ma.masked_array(autos, mask)

    return masked_autos


def hyperfine_delay(autos, high_ind):
    nfreqs = autos.shape[-2]
    window = signal.blackmanharris(nfreqs)
    kappa = np.arange(high_ind - 1, high_ind + 1, 0.0005)
    kappa = kappa[:, np.newaxis]
    i = np.arange(nfreqs)
    hyperfine_delay = (
        np.abs(autos) * window * np.exp(-1j * 2 * np.pi * kappa * i / nfreqs)
    )
    hyperfine_delay = np.sum(hyperfine_delay, axis=-1)


def fit_cable_reflections(uv, data, init_delays, polyfit=True):

    reflection_coeffs = np.empty((3, uv.Nants_data, data.shape[-1]))

    # check that using autos
    if np.any(uv.ant_1_array != uv.ant_2_array):
        raise ValueError("can only use autos to fit cable reflections")

    # work out future array shapes

    # split nblts into ntimes, nautos if not already done
    # if uv.Ntimes != 1:
    #     if uv.data_array.shape[0] == uv.Nblts:
    #         uv.data_array = uv.data_array.reshape(
    #             uv.Ntimes, uv.Nants_data, uv.Nfreqs, uv.Npols
    #         )

    # compute delay transform
    if uv.future_array_shapes:
        dv = uv.channel_width[0] / 1e6
        v = uv.freq_array / 1e6
    else:
        dv = uv.channel_width / 1e6
        v = uv.freq_array[0, :] / 1e6

    N = uv.Nfreqs
    dt = 1 / (N * dv)
    print("delay resolution (microseconds): " + str(dt))
    amp = dv * N
    delay_array = np.arange(0, N * dt, dt)

    data_delay = np.fft.ifft(data.real, axis=-2)
    data_delay = np.swapaxes(data_delay, -2, -1)
    data_delay = data_delay * amp * np.exp(1j * v[0] * 2 * np.pi * delay_array)
    data_delay = np.swapaxes(data_delay, -2, -1)

    data_delay = data_delay[..., 0 : int(N / 2), :]
    delay_array = delay_array[0 : int(N / 2)]

    for i in range(uv.Nants_data):
        for j in range(data.shape[-1]):
            # get initial guesses for amplitude, tau, and phase
            high_ind = (np.abs(delay_array - init_delays[i])).argmin()
            print("initial index of tau: " + str(high_ind))
            phase0 = np.angle(data_delay[i, high_ind, j])
            tau0 = delay_array[high_ind]
            print("initial approximate phase: " + str(phase0))
            print("initial approximate tau: " + str(tau0))
            amp0 = np.abs(data_delay[i, high_ind, j]) / np.abs(data_delay[i, 0, j])
            print("initial approximate amplitude: " + str(amp0))

            # perform hyperfine transform
            print("hyperfine delay resolution (microseconds): " + str(dt / 2000))
            window = signal.windows.blackmanharris(N)
            kappa = np.arange(high_ind - 2, high_ind + 2, 0.0005)
            kappa = kappa[:, np.newaxis]
            k = np.arange(N)
            hyperfine_delay = (
                data.real[..., i, :, j]
                * window
                * np.exp(-1j * 2 * np.pi * kappa * k / N)
            )
            hyperfine_delay = np.sum(hyperfine_delay, axis=-1)
            delay_ind = np.where(
                np.abs(hyperfine_delay) == np.max(np.abs(hyperfine_delay))
            )[0][0]
            tau1 = delay_array[high_ind - 2] + dt * delay_ind / 2000
            print("approximate tau: " + str(tau1))
            amp1 = np.abs(hyperfine_delay)[delay_ind] / np.sum(
                np.abs(data.real[..., i, :, j]) * window
            )
            print("approximate amplitude: " + str(amp1))

            if polyfit:
                # perform a polynomial fit
                out = poly.Polynomial.fit(v, data.real[..., i, :, j], deg=7)
                coeff = out.convert().coef
                # find best phase guess
                def f(bp, phi):
                    return poly.polyval(bp, coeff) * (
                        1 + (amp1 ** 2) + 2 * amp1 * np.cos(2 * np.pi * tau1 * bp + phi)
                    )

            else:

                def f(bp, phi):
                    return (
                        1 + (amp1 ** 2) + 2 * amp1 * np.cos(2 * np.pi * tau1 * bp + phi)
                    )

            popt, pcov = curve_fit(f, v, data.real[..., i, :, j], p0=phase0)
            phase1 = popt[0]
            print("approximate phase: " + str(phase1))

            reflection_coeffs[:, i, j] = tau1, amp1, phase1
            if i == 0 and j == 0:
                print(reflection_coeffs[:, i, j])
                print(tau1, amp1, phase1)
    # fit cable reflections
    # return coefficients
    return reflection_coeffs


def apply_cable_reflections(uv, data, reflection_coeffs, return_terms=True):

    if uv.future_array_shapes:
        v = uv.freq_array / 1e6
    else:
        v = uv.freq_array[0, :] / 1e6
    tau = reflection_coeffs[0, ...][:, np.newaxis, :]
    amp = reflection_coeffs[1, ...][:, np.newaxis, :]
    phase = reflection_coeffs[2, ...][:, np.newaxis, :]
    reflection_terms = (
        1 + (amp ** 2) + 2 * amp * np.cos(2 * np.pi * tau * v[:, np.newaxis] + phase)
    )

    cable_autos = np.copy(data)
    cable_autos /= reflection_terms

    if return_terms:
        return cable_autos, reflection_terms


def compute_delay_spectra(uv, data, label, mask=None, bgains=False, bh=True):
    data_product = {"label": label}

    if uv.future_array_shapes:
        dv = uv.channel_width[0] / 1e6
        v = uv.freq_array / 1e6
    else:
        dv = uv.channel_width / 1e6
        v = uv.freq_array[0, :] / 1e6

    N = uv.Nfreqs
    if bgains:
        N = int(N * 2 / 3)
    dt = 1 / (N * dv)
    print("delay resolution (microseconds): " + str(dt))
    amp = dv * N
    delay_array = np.arange(0, N * dt, dt)

    data_product["freq_data"] = np.copy(data)

    if uv.future_array_shapes:
        data_product["v1"] = uv.freq_array / 1e6
    else:
        data_product["v1"] = uv.freq_array[0, :] / 1e6
    # data_product["v1"] = np.copy(uv.freq_array[0, 0:N] / 1e6)

    # inds = np.where(mask)[0]
    # # uv2.data_array.real[:, 0, inds, :] = 0.0
    # uv2.flag_array[:, 0, inds, :] = True
    # # uv2.nsample_array[:, 0, inds, :] = 0

    # broad_mask = np.broadcast_to(mask[0:N, np.newaxis], (uv2.Nblts, N, 4))
    # data_product["mask_data"] = np.ma.masked_array(
    #     uv2.data_array.real[:, 0, 0:N, :], broad_mask
    # )

    # uv2.frequency_average(avg)

    # if bgains:
    #     N = int(512 / avg)
    # else:
    #     N = int(768 / avg)

    # data_product["av_data"] = np.copy(uv2.data_array.real[:, 0, 0:N, :])
    # data_product["v2"] = np.copy(uv2.freq_array[0, 0:N] / 1e6)

    # dv = 0.04 * avg
    # dt = 1 / (N * dv)
    # amp = dv * N

    # average coarse band shape
    n_fine_chans = N / 24
    cb_av = np.full(int(n_fine_chans), 1)
    cb_av[0] = 0
    cb_av[-1] = 0
    cb_av = np.tile(cb_av, 24)

    cb_delay = np.fft.ifft(cb_av[0:N])
    cb_delay = cb_delay[0 : int(N / 2)]
    cb_lines = np.where(np.abs(cb_delay) > 1e-5)[0]

    data_product["cb_lines"] = cb_lines
    # apply blackman-harris
    if bh:
        window = signal.blackmanharris(N)
        data *= window[:, np.newaxis]
        wnorm = N / np.sum(window)
        amp *= wnorm

    delay_array = np.arange(0, N * dt, dt)
    delay_array_crop = delay_array[0 : int(N / 2)]
    data_product["delay_array_crop"] = delay_array_crop
    data_product["bh_data"] = np.copy(data)

    # mask = uv2.flag_array[0, 0, 0:N, :]
    # broad_mask = np.broadcast_to(mask, (uv2.Nblts, N, 4))
    # data = uv2.data_array.real[:, 0, 0:N, :]
    # masked_data = np.ma.masked_array(data, broad_mask)
    # data_delay = np.fft.ifft(masked_data, axis=1)
    # data_delay = np.swapaxes(data_delay, 1, 2)
    # data_delay = (
    #     data_delay * amp * np.exp(-1j * data_product["v2"][0] * 2 * np.pi * delay_array)
    # )
    # data_delay = np.swapaxes(data_delay, 1, 2)

    data_delay = np.fft.ifft(data, axis=-2)
    data_delay = np.swapaxes(data_delay, -1, -2)
    data_delay = (
        data_delay * amp * np.exp(-1j * data_product["v1"][0] * 2 * np.pi * delay_array)
    )
    data_delay = np.swapaxes(data_delay, -1, -2)
    # data_delay = data_delay.reshape(uv.Ntimes, uv.Nbls, N, uv.Npols)
    data_product["data_delay_crop"] = data_delay[:, :, 0 : int(N / 2), :]
    data_product["time_av_data_delay"] = np.mean(
        data_product["data_delay_crop"], axis=0
    )
    data_product["time_std_data_delay"] = np.std(
        data_product["data_delay_crop"], axis=0
    )

    return data_product

