import librosa
import numpy as np


def afilter(wave, f, threshold=5):
    t1 = max(abs(wave))*(threshold/100)
    wave[(abs(wave) <= t1)] = 0
    return wave


def stdft(wave, f, wl, step, scale=True, _complex_=False):
    W = np.hanning(wl)
    def comp(x): return np.fft.fft(wave[int(x):int(wl+x)]*W)
    z = np.array([comp(i) for i in step])
    z = z[0:(wl//2), :]
    z = z/wl  # scaling by the original number of fft points

    if (_complex_ == False):
        z = 2*abs(z)  # multiplied by 2 to save the total energy see http://www.ni.com/white-paper/4278/en/, section Converting from a Two-Sided Power Spectrum to a Single-Sided Power Spectrum
        if (scale):
            z = z/z.max()  # normalise to 1 the complete matrix
    return z


def fund(wave, f, wl=512, ovlp=0, threshold=0, fmax=None):
    WL = wl//2

    if (not fmax):
        fmax = f/2

    # THRESHOLD
    if (threshold):
        wave = afilter(wave=wave, f=f, threshold=threshold)

    # SLIDING WINDOW
    wave[wave == 0] = 1e-06
    n = len(wave)
    step = np.arange(0, n-wl, wl-(ovlp * wl/100))
    N = len(step)
    z1 = np.zeros((wl, N))
    YY1 = np.zeros((wl, N))

    for i in step:
        yy1 = abs(np.fft.fft(wave[int(i):int(wl + i)]))
        yy2 = ((np.fft.ifft(np.log(yy1))).real)
        z1[:, step == i] = yy2.reshape(len(yy2), 1)
        YY1[:, step == i] = yy1.reshape(len(yy1), 1)

    z2 = z1[0:WL, :]
    z = z2
    z[np.isnan(z) | np.isneginf(z)] = 0
    fmaxi = f//fmax

    tfund = np.zeros((N,), dtype='float64')
    for k in range(N):
        tfund[k] = np.argmax(z[fmaxi:, k])
    tfund[tfund == 0] = float('nan')
    ffund = f/(tfund + fmaxi)
    y = ffund/1000
    return y


def dfreq(wave, f, wl=512, ovlp=0, threshold=0, bandpass=None, fftw=False):

    if (threshold != 0):
        wave = afilter(wave=wave, f=f, threshold=threshold)

    n = len(wave)
    step = np.arange(0, n-wl, wl-(ovlp * wl/100))

    # Fourier
    step = np.round(step)
    y1 = stdft(wave=wave, f=f, wl=wl, step=step)

    if (bandpass):
        lowlimit = np.round((wl*bandpass[0])/f)
        upperlimit = np.round((wl*bandpass[1])/f)
        y1[:, :int(lowlimit)] = 0
        y1[:, int(upperlimit):] = 0

    # Maximum search
    maxi = np.max(y1, axis=1)
    y2 = np.argmax(y1, axis=1).astype('float32')
    y2[(maxi == 0)] = float('nan')

    # converts into frequency
    y = (f*y2)/(1000*wl)
    return y


def sfm(spec):
    if (len(spec) > 4000):
        step = np.arange(1, len(spec), round(len(spec)/256))
        spec = spec[step]

    spec[spec == 0] = 1e-5
    n = len(spec)
    geo = np.prod(np.power(spec, 1/n))
    ari = np.mean(spec)
    flat = geo/ari
    return flat


def sh(spec, alpha='shannon'):
    N = len(spec)
    spec[spec == 0] = 1e-7
    spec = spec/sum(spec)
    if (alpha == 'shannon'):
        z = -sum(spec*np.log(spec))/np.log(N)
    elif (alpha == 'simpson'):
        z = 1 - sum(spec ^ 2)
    else:
        if (alpha < 0):
            print("'alpha' cannot be negative.")
            return
        if (alpha == 1):
            print("'alpha' cannot be set to 1.")
            return
        z = (1/(1-alpha))*np.log(sum(np.power(spec, alpha)))/np.log(N)

    return z


def feature_extractor(file_name):
    audio, sr = librosa.load(file_name, sr=48000)

    n = len(audio)
    w = np.hanning(n)
    y = audio*w
    y = abs(np.fft.fft(y))
    y = 2*y[0:(n//2)]
    y = y/max(y)

    x = (np.arange(0, n)*(sr/(n*1000)))[1:(n//2)]
    x = x * 1000

    l = len(y)
    wl = l * 2

    y = y[0:int(280 * wl/sr)]
    #
    l = len(y)

    shift = 280/l
    freq = np.arange(0, 280+(shift/2))
    freq = x[x <= 280]/1000

    # Amplitude
    amp = y/sum(y)
    cumamp = np.cumsum(amp)

    mean_freq = sum(amp*freq)
    sd = np.sqrt(sum(amp*(np.square(freq-mean_freq))))
    median = freq[len(cumamp[cumamp <= 0.5])+1]
    mode = freq[np.argmax(amp)]
    Q25 = freq[len(cumamp[cumamp <= 0.25])+1]
    Q75 = freq[len(cumamp[cumamp <= 0.75])+1]
    IQR = Q75 - Q25
    cent = sum(freq * amp)
    z = amp - np.mean(amp)
    w = np.std(amp)
    skew = (sum(np.power(z, 3))/(l-1))/np.power(w, 3)
    kurt = (sum(np.power(z, 4)/(l-1)))/np.power(w, 4)
    sfm_amp = sfm(amp)
    sh_amp = sh(amp)
    # prec = sr/wl
    WL = 2048
    threshold = 5
    ff = fund(wave=audio, f=sr, ovlp=50, threshold=threshold, fmax=280, wl=WL)
    meanfun = np.nanmean(ff)
    minfun = np.nanmin(ff)
    maxfun = np.nanmax(ff)

    ff = dfreq(wave=audio, f=sr, wl=WL, ovlp=0, bandpass=[
               0, 20000], threshold=threshold, fftw=True)
    meandom = np.nanmean(ff)
    mindom = np.nanmin(ff)
    maxdom = np.nanmax(ff)
    dfrange = (maxdom - mindom)

    changes = abs(ff[:-1] - ff[1:])
    if (mindom == maxdom):
        modindx = 0
    else:
        modindx = np.nanmean(changes)/dfrange

    return np.array((mean_freq, sd, median, Q25, Q75, IQR, skew, kurt, sh_amp, sfm_amp, mode, cent, meanfun, minfun, maxfun, meandom, mindom, maxdom, dfrange, modindx))
