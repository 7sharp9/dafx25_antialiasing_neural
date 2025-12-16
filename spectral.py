import time

from scipy.signal.windows import chebwin
import scipy
from scipy.signal import firwin, freqz, kaiser_beta
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

def cheb_fft(x, nfft=None, dim=-1, at=-120, return_window=False):
    '''
    Chebyshev windowed and take FFT
    :param x: signal
    :param nfft: FFT length
    :param dim: time dimension
    :param at: Chebyshev window side-lobe attenuation [dB]
    :param return_window: if True, returns tuple of (FFT, window), otherwise returns FFT. Default: False.
    :return: (FFT, window) if return_window=True, otherwise returns FFT
    '''
    win = torch.tensor(chebwin(x.shape[dim], at=at, sym=False), device=x.device).double()
    dims = [1] * x.ndim
    dims[dim] = -1
    win = win.view(dims)
    if nfft is None:
        nfft = x.shape[dim]
    X = torch.fft.rfft(x * win, nfft)
    if return_window:
        return X, win
    else:
        return X

def bandlimit_signal(sig, fs, f0, IS_SYM=False, APPLY_LP=False, PLOT_SIG=False, cheb_at=-120):

    f0 = f0.to(torch.double)
    # truncate signal to remove transient and leave one-second fragment
    if len(sig) > fs:
        sig = sig[-fs:]
    N = len(sig)

    # check for odd length
    if N % 2 != 0:
        print('Signal should have an even length after truncation!')
    assert N % 2 == 0, 'Signal should have an even length after truncation!'

    # calculate harmonics
    num_harmonics = int(0.5 * fs / f0)
    if IS_SYM:
        harmonics = f0 * torch.arange(1, 2 * num_harmonics + 1, 2, device=sig.device)
    else:
        harmonics = f0 * torch.arange(1, num_harmonics + 1, device=sig.device)

    S, win = cheb_fft(sig, at=cheb_at, return_window=True)

    # adjust for scalloping loss (including window)
    bins_exact = (harmonics * N / fs)  # exact bins for harmonics
    bins = torch.round(bins_exact).to(torch.int64)  # discrete bins for harmonics
    d = bins_exact - bins  # difference between exact and discrete bins
    idx = torch.arange(N, device=sig.device).double()  # indexes for summation

    # DC bin
    dc = torch.real(S[0]) / torch.sum(win)

    # synthesize bandlimited signal
    t = torch.arange(N, dtype=sig.dtype, device=sig.device) / fs
    complex_amps = S[bins] / torch.sum(win.view(-1, 1) * torch.exp(1j * 2 * torch.pi * d * idx.view(-1, 1) / N), dim=0)
    amp = complex_amps.abs()
    phase = complex_amps.angle()
    partials = 2 * amp * torch.cos(2 * torch.pi * harmonics * t.view(-1, 1) + phase)
    sig_lim = dc + torch.sum(partials, -1)

    # calculate aliased components
    alias = sig - sig_lim

    # debug plots
    if PLOT_SIG:
        plt.figure()
        plt.plot(sig.detach().numpy(), color="#e6194b", label="Input signal")
        plt.plot(sig_lim.detach().numpy(), color="#3cb44b", label="Bandlimited signal", linestyle='--')
        plt.plot(alias.detach().numpy(), color="#4363d8", label="Alias signal")
        plt.xlim([-100, len(sig) + 100])
        plt.xlabel('Samples [n]', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.legend()
        #plt.title(f"Input signal at $f_0 = {f0:.2f}$ Hz", fontsize=14)
        plt.grid(True)
        plt.xlim([0, 5*fs/f0])
        plt.show()



    return sig, sig_lim, alias, complex_amps

def bandlimit_batch(sig, f0, fs, cheb_at=-120):
    '''
    Bandlimit (remove aliasing) from a batch of harmonic signals
    :param sig: signals to band-limit of dims (batch, time)
    :param f0: fundamental frequency of signals of dim (batch) [Hz]
    :param fs: sample rate [Hz]
    :param cheb_at: Chebyshev window side-lobe attenuation [dB]. Default=-120
    :return: tuple of (bandlimited signal, aliases)
    '''
    f0 = f0.to(torch.double)

    if sig.ndim == 3:
        sig = sig.squeeze()
    assert sig.ndim == 2

    # truncate signal to remove transient and leave one-second fragment
    if sig.shape[-1] > fs:
        sig = sig[..., -fs:]
    N = sig.shape[-1]

    assert N % 2 == 0, 'Signal should have an even length after truncation!'

    S, win = cheb_fft(sig, at=cheb_at, return_window=True)
    sig_lims = synthesise_batch(S, win, fs, f0)
    aliases = sig - sig_lims

    return sig_lims, aliases

@torch.jit.script
def synthesise_batch(spectrum: torch.Tensor,
                     win: torch.Tensor,
                     fs: int,
                     f0: torch.Tensor):
    '''
    Bandlimit (remove aliasing) from a batch of harmonic spectra
    :param spectrum: spectra of dims (batch, bins)
    :param win: window used to take FFT
    :param fs: sample rate [Hz]
    :param f0: fundamental frequency of signals of dim (batch) [Hz]
    :return: bandlimited signals (batch, time)
    '''
    win_sum = torch.sum(win)
    batch_size = spectrum.shape[0]
    N = win.shape[-1]
    time_idx = torch.arange(N, device=win.device).double()
    sig_lims = win.new_zeros(batch_size, N)


    # iterate through batch
    for b in range(batch_size):
        num_harmonics = int(torch.floor(0.5 * fs / f0[b, ...]))
        harmonics = f0[b, ...] * torch.arange(1, num_harmonics + 1, device=spectrum.device)

        # adjust for scalloping loss (including window)
        bins_exact = (harmonics * N / fs)
        bins = torch.round(bins_exact).to(torch.int64)  # discrete bins for harmonics
        d = bins_exact - bins                           # difference between exact and discrete bins

        # DC bin
        dc = torch.real(spectrum[b, 0]) / win_sum

        complex_amps = spectrum[b, bins] / torch.sum(win.view(-1, 1) * torch.exp(1j * 2 * torch.pi * d * time_idx.view(-1, 1) / N), dim=0)
        amp = complex_amps.abs()
        phase = complex_amps.angle()
        partials = 2 * amp * torch.cos(2 * torch.pi * harmonics * time_idx.view(-1, 1) / fs + phase)
        sig_lim = dc + torch.sum(partials, -1)
        sig_lims[b, :] = sig_lim
        #sig_lims.append(sig_lim)

    return sig_lims


class PerceptualFIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module. CODE AND DOCS ADAPTED FROM: https://github.com/csteinmetz1/auraloss

    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False

    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    A-weighting filter - "aw"
    First-order highpass - "hp"
    Folded differentiator - "fd"
    Loss pass filter - "lp"

    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False, double_precision=True):
        """Initilize FIR pre-emphasis filtering module."""
        super(PerceptualFIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot
        self.double_precision = double_precision

        import scipy.signal

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "lp":
            As = 80
            pb_edge = 12e3
            sb_edge = 16e3
            delta_f = (sb_edge - pb_edge) / fs
            fc = (sb_edge + pb_edge) / 2
            N = int(np.ceil((As - 7.95) / 14.36 / delta_f))
            N += (N % 2)
            h = firwin(N + 1, cutoff=fc, fs=fs, window=('kaiser', kaiser_beta(As)))

            self.ntaps = N+1
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=N+1, bias=False, padding='same')
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(h).view(1, 1, -1)

        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding='same'
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float64")).view(1, 1, -1)


    def forward(self, x):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        if x.ndim > 2:
            x = x.squeeze()

        if x.ndim == 2:
            x = x.unsqueeze(1)

        if not self.double_precision:
            taps = self.fir.weight.data.to(torch.float32)
        else:
            taps = self.fir.weight.data
        x = torch.nn.functional.conv1d(
            x, taps, padding=self.ntaps // 2
        ).squeeze()
        return x





if __name__ == '__main__':

    B = 32
    x = torch.rand(B, 44100) - 0.5
    f0 = 1000 * (torch.rand(B, 1) + 1.0)

    y = bandlimit_batch(x, f0, 44100)

    start_time = time.perf_counter()
    for n in range(500):
        y = bandlimit_batch(x, f0, 44100)
    print(time.perf_counter() - start_time)
    #
    # As = 80
    # fs = 44100
    # pb_edge = 12e3
    # sb_edge = 16e3
    # delta_f = (sb_edge - pb_edge) / fs
    # fc = (sb_edge + pb_edge) / 2
    # N = np.ceil((As - 7.95) / 14.36 / delta_f)
    # N += (N % 2)
    # h = firwin(N+1, cutoff=fc, fs=fs, window=('kaiser', kaiser_beta(As)))
    # w, H = freqz(h, fs=fs)
    #
    # mpl.use('macosx')
    # # plt.plot(h)
    # # plt.show()
    #
    # plt.plot(w, 20 * np.log10(np.abs(H)))
    # plt.show()
#     # sr = 44100
#     # dur = 0.2
#     #
#     # Nfft = 44100
#     #
#     # f0 = torch.DoubleTensor([1245])
#     # t_ax = torch.arange(0, int(dur * sr), dtype=torch.double) / sr
#     # x = torch.tanh(10*(0.5 + torch.sin(2*torch.pi*f0 * t_ax)))
#     # x, x_bl, _, _ = bandlimit_signal(x, sr, f0)
#     #
#     # X = cheb_fft(x, nfft=Nfft)
#     # norm = X.abs().max()
#     # X /= norm
#     #
#     #
#     # X_bl = cheb_fft(x_bl, nfft=Nfft)
#     # X_bl /= norm
#     #
#     #
#     # # snra = 10 * torch.log10(
#     # #     torch.sum(x_bl ** 2) / torch.sum((x - x_bl)**2)
#     # # )
#     # #print('SNRA = ', snra)
#     #
#     # plt.plot(20 * torch.log10(X.abs()))
#     # plt.plot(20 * torch.log10(X_bl.abs()))
#     #
#     # # plt.plot(x)
#     # # plt.plot(x_bl)
#     # # plt.plot(x - x_bl)
#     # plt.show()
#
#     # Batch spectral
#
#
#     mpl.use('macosx')
#     sr = 44100
#     dur = 1.2
#
#     Nfft = 44100
#
#     f0 = torch.DoubleTensor([1245, 4186]).view(-1, 1)
#     t_ax = torch.arange(0, int(dur * sr), dtype=torch.double) / sr
#     x = torch.tanh(10 * (0.5 + torch.sin(2 * torch.pi * f0 * t_ax)))
#
#     y = bandlimit_signal_batched(x, sr, f0)
#     print(x)

