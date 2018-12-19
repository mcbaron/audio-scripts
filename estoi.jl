#estoi.jl
function estoi(refSig, inSig, fs_signal)
# Implementation of the Extended Short-Time Objective Intelligibility (ESTOI) predictor
#   d = estoi(ref, in, fs) returns the output of the extended short-time
#   objective intelligibility (ESTOI) predictor.
#
# Implementation of the Extended Short-Time Objective
# Intelligibility (ESTOI) predictor, described in Jesper Jensen and
# Cees H. Taal, "An Algorithm for Predicting the Intelligibility of
# Speech Masked by Modulated Noise Maskers," IEEE Transactions on
# Audio, Speech and Language Processing, 2016.
#
# Input:
#        ref: clean reference time domain signal
#        in:  noisy/processed time domain signal
#        fs:  sampling rate [Hz]
#
# Output:
#        d: intelligibility index

    if length(refSig) != length(inSig)
        error("The clean and processed speech are not of the same length")

    end

    # initialization
    refSig = vec(refSig) # clean speech column vector
    inSig = vec(inSig) # processed speech column vector

    fs = 10000 # sample rate of proposed intelligibility measure
    N_frame	= 256  # window support
    overlap = div(N_frame, 2)
    K = 512 # FFT size
    J = 15  # Number of 1/3 octave bands
    mn = 150  # Center frequency of first 1/3 octave band in Hz.
    H = thirdoct(fs, K, J, mn) # Get 1/3 octave band matrix
    N = 30  # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta = -15 # lower SDR-bound
    dyn_range = 40  # speech dynamic range

    # resample signals if other samplerate is used than fs
    if fs_signal != fs
        src2 = convert(Float64, fs/fs_signal)
        refSign = resample(refSig, src2)
        inSig = resample(inSig, src2)
    end

    # remove silent frames
    x, y = removeSilentFrames(refSig, inSig, dyn_range, N_frame, overlap)

    # apply 1/3 octave band TF-decomposition
    x_hat = stft(x, N_frame, overlap, nfft=K, window=hanning) # apply short-time DFT to clean speech
    y_hat = stft(y, N_frame, overlap, nfft=K, window=hanning) # apply short-time DFT to processed speech

    X = fill(0.0, (J, size(x_hat, 2))) # init memory for clean speech 1/3 octave band TF-representation
    Y = fill(0.0, (J, size(y_hat, 2))) # init memory for processed speech 1/3 octave band TF-representation
    for j = 1:J
        active_bank = abs2.(freqz(H[j], range(0, length=size(x_hat,1), stop=pi)))
        for i = 1:size(x_hat, 2)
            X[j, i]	= sum(sqrt.(active_bank.*abs2.(x_hat[:, i]))) # apply 1/3 octave band filtering
            Y[j, i]	= sum(sqrt.(active_bank.*abs2.(y_hat[:, i])))
        end
    end

    # loop all segments of length N and obtain intermediate intelligibility measure for each
    d1 = fill(0.0,(1+size(X, 2)-N)) # init memory for intermediate intelligibility measure
    for m in N:size(X,2)
        X_seg = X[:, (m-N+1):m] # region of length N with clean TF-units for all j
        Y_seg = Y[:, (m-N+1):m] # region of length N with processed TF-units for all j
        X_seg = X_seg + fill(eps(), size(X_seg)) # to avoid divide by zero
        Y_seg = Y_seg + fill(eps(), size(Y_seg)) # to avoid divide by zero

        ## first normalize rows (to give \bar{S}_m)
        XX = X_seg .- mean(X_seg, dims=2) # normalize rows to zero mean
        YY = Y_seg .- mean(Y_seg, dims=2) # normalize rows to zero mean

        YY = Diagonal(1 ./ sqrt.(Diagonal(YY*YY')))*YY # normalize rows to unit length
        XX = Diagonal(1 ./ sqrt.(Diagonal(XX*XX')))*XX # normalize rows to unit length

        XX = XX + fill(eps(), (size(XX))) # to avoid corr.div.by.0
        YY = YY + fill(eps(), (size(YY))) # to avoid corr.div.by.0

        ## then normalize columns (to give \check{S}_m)
        YYY = YY .- mean(YY, dims=1) # normalize cols to zero mean
        XXX = XX .- mean(XX, dims=1) # normalize cols to zero mean

        YYY = YYY*Diagonal(1 ./ sqrt.(Diagonal(YYY'*YYY))) # normalize cols to unit length
        XXX = XXX*Diagonal(1 ./ sqrt.(Diagonal(XXX'*XXX))) # normalize cols to unit length

        #compute average of col.correlations (by stacking cols)
        d1[m-N+1] = 1/N * vec(XXX)'*vec(YYY)
    end

    return d = mean(d1)
end



##
function thirdoct(fs, N_fft, numBands, mn)
#   THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
#   inputs:
#       FS:         samplerate
#       N_FFT:      FFT size
#       NUMBANDS:   number of bands
#       MN:         center frequency of first 1/3 octave band
#   outputs:
#       A:          octave band matrix

    f = range(0, stop=fs/2, length = convert(Int, N_fft/2 + 1)) |> collect
    k = 0:(numBands-1)
    cf = 2 .^(k/3)*mn
    fl = sqrt.((2 .^(k./3)*mn).*2 .^((k.-1)./3)*mn)
    fr = sqrt.((2 .^(k./3)*mn).*2 .^((k.+1)./3)*mn)
    A = Array{ZeroPoleGain}(undef,numBands)
    prototype = Butterworth(10)

    for i = 1:numBands
        responsetype = Bandpass(fl[i], fr[i], fs=fs)
        A[i] = digitalfilter(responsetype, prototype)
    end

    # Plot filterbank
    # fig = plot(show=false)
    # for i = 1:numBands
    #     ω = range(0, length=256, stop=pi)
    #     H = freqz(A[i], ω)
    #     plot!(log.(fs/2 * ω/pi), 20*log.(10, abs2.(H)))
    # end
    # display(fig)

    return A

end


##
function removeSilentFrames(x, y, dyn_range, N, K)
#   REMOVESILENTFRAMES(X, Y, DYN_RANGE, N, K) X and Y
#   are segmented with frame-length N and overlap K, where the maximum energy
#   of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
#   reconstructed signals, excluding the frames, where the energy of a frame
#   of X is smaller than X_MAX-RANGE

    x = vec(x)
    y = vec(y)
    frames = 1:K:(length(x)-N)

    w = hanning(N)
    msk = fill(0.0, size(frames))

    for j = 1:length(frames)
        jj = convert(Int,frames[j]):convert(Int,(frames[j]+N-1))
        msk[j] = 20*log10(norm(x[jj].*w)./sqrt(N))
    end

    msk = (msk.-maximum(msk).+dyn_range).>0
    count = 1

    x_sil = fill(0.0, size(x))
    y_sil = fill(0.0, size(y))

    for j = 1:length(frames)
        if msk[j]
            jj_i = frames[j]:(frames[j]+N-1)
            jj_o = frames[count]:(frames[count]+N-1)
            x_sil[jj_o] = x_sil[jj_o] + x[jj_i].*w
            y_sil[jj_o] = y_sil[jj_o] + y[jj_i].*w
            count += 1
        end
        global jj_o
    end

    x_sil = x_sil[1:jj_o[end]]
    y_sil = y_sil[1:jj_o[end]]

    return x_sil, y_sil

end
