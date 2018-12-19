module ffSNR

using WAV:wavread
using DSP

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    println("Fodor & Fingscheidt SNR")
    println("$(ARGS[1])")

    # Fodor & Fingscheidt Reference-free SNR Measurement

    noisySpeech, fs = wavread(ARGS[1])
    noisySpeech = .5 * sum(noisySpeech, dims=2)

    # Need FS to be either 8 or 16 kHz
    if fs > 16e3
        desiredfs = 16e3
    elseif fs < 8e3
        desiredfs = 8e3
    else
        desiredfs = fs
    end
    noisySpeech = resample(vec(noisySpeech), convert(Float64, desiredfs/fs))
    fs = desiredfs

    stepSize = 1.625e-5*(fs - 8e3) + 1.07
    Ltrans = convert(Int, 5 + fld(fs, 8e3))
    alphaSPD = 10^(Ltrans+1)
    gammaSPD = -3.75e-6 * (fs - 8e3) + 1.085

    winlength = convert(Int, .032*fs)
    # STFT coefficients, 20ms windows 50% overlap, hann windows
    nSpeechStftCoeff = stft(noisySpeech[:], winlength, div(winlength, 2); onesided=eltype(noisySpeech)<:Real,
                        nfft=nextfastfft(winlength), window=hanning)

    K, L = size(nSpeechStftCoeff)
    stftFreqs = rfftfreq(nextfastfft(winlength), fs)
    speechFreqs = (stftFreqs .≥ 500) .& (stftFreqs .≤ 2500)
    scriptK = sum(speechFreqs)

    # Note: time indexes columns
    # nSpeechStftCoeffSmoothed = zeros(Complex, size(nSpeechStftCoeff))
    nSpeechStftCoeffSmoothed = repeat(.5*nSpeechStftCoeff[:,1], 1, L)
    for l = 2:L
        nSpeechStftCoeffSmoothed[:, l] = sqrt.(.5*abs2.(nSpeechStftCoeff[:,l]) + .5*abs2.(nSpeechStftCoeffSmoothed[:,l-1]))
    end

    dynThresh = fill(5e1, K, L)
    adpThresh = fill(1.0, 1, L)
    controlSPD = fill(5e1, 1, L)
    floorSignal = fill(0.0, 1, L)
    Py = fill(0.0, 1, L)
    H_SPD = fill(0, 1, L)
    Hsp = fill(0.0,K,L)
    Hsa = fill(0.0,K,L)
    Hst = fill(0.0,K,L)

    # Noise Variance Tracking
    noiseVariance = zeros(size(nSpeechStftCoeff))
    frameSP = falses(1,L) # Frame membership in Λ1
    frameSA = falses(1,L) # Frame membership in Λ0
    for l = 2:L
        # Find appropriate hypothesis for each frame
        Hsp[:,l] = abs2.(nSpeechStftCoeffSmoothed[:,l]) .> 2*dynThresh[:,l-1]

        Hsa[:,l] = (abs2.(nSpeechStftCoeffSmoothed[:,l]) .≤ 2*dynThresh[:,l-1]) .&
                    (abs2.(nSpeechStftCoeffSmoothed[:,l]) .< noiseVariance[:,l-1])

        Hst[:,l] = (abs2.(nSpeechStftCoeffSmoothed[:,l]) .≤ 2*dynThresh[:,l-1]) .&
                    (abs2.(nSpeechStftCoeffSmoothed[:,l]) .≥ noiseVariance[:,l-1])

        smoothing = 1.0*Hsp[:,l] + .5*Hsa[:,l] + .875*Hst[:,l]
        # update noiseVariance
        noiseVariance[:,l] = smoothing .* noiseVariance[:,l-1] +
                            (1.0 .- smoothing) .* abs2.(nSpeechStftCoeffSmoothed[:,l])

        # update dynamic threshold
        a = BitArray(Hsp[:,l])
        b = abs2.(nSpeechStftCoeffSmoothed[:,l]) .< dynThresh[:,l-1]
        dynThresh[:,l] = dynThresh[:,l-1] # else clause
        dynThresh[a,l] *= stepSize
        dynThresh[b,l] = abs2.(nSpeechStftCoeffSmoothed[b,l-1])

        # Frame-wise Voice Activity Detection
        activeFrame = sum(Hsp[speechFreqs,l] + Hst[speechFreqs,l]) / scriptK
        frameSP[l] = activeFrame ≥ .9


        # Speech Pause Detection
        Py[l] = (1/scriptK) * sum(abs2.(nSpeechStftCoeffSmoothed[speechFreqs, l]))

        # update adpThresh
        adpThresh[l] = 5*floorSignal[l-1] + alphaSPD

        # Pick appropriate hypothesis
        LtransFrames = max(1, l-Ltrans)
        if Py[l] > adpThresh[l]
            H_SPD[l] = 2 # Speech Presence
        elseif any(H_SPD[LtransFrames:l] == 2)
            H_SPD[l] = 1 # Speech Transition
        else
            H_SPD[l] = 0 # Speech Absent
        end

        frameSA[l] = H_SPD[l] == 0

        # calculate betaSPD
        scriptA = (Py[l] .≤ 2 * controlSPD[l-1]) .& (H_SPD[l] != 1)
        if scriptA && Py[l] > floorSignal[l-1]
            betaSPD = .875
        elseif Py[l] ≤ floorSignal[l-1] && scriptA
            betaSPD = .5
        else
            betaSPD = 1
        end

        # update floorSignal
        floorSignal[l] = betaSPD * floorSignal[l-1] + (1-betaSPD) * Py[l]

        # update controlSPD
        controlSPD[l] = controlSPD[l-1] # else clause
        if Py[l] > 2 * controlSPD[l-1]
            controlSPD[l] *= gammaSPD
        elseif Py[l] < controlSPD[l-1]
            controlSPD[l] = Py[l]
        end

    end

    # Since these parameters need time to adapt, let's drop the first 15 frames
    nSpeechStftCoeff = nSpeechStftCoeff[:,15:end]
    noiseVariance = noiseVariance[:,15:end]
    frameSP = frameSP[15:end]
    frameSA = frameSA[15:end]


    # Now we can calculate the
    card0 = sum(frameSA)
    card1 = sum(frameSP)
    Ps = (1/(K*card1)) * sum(max.(0, abs2.(nSpeechStftCoeff[:,frameSP]) - noiseVariance[:,frameSP]))
    Pn = (1/(K*card0)) * sum(noiseVariance[:,frameSA])

    rawSNR = 10*log(10, Ps / Pn)

    # correct the estimate
    # Magic numbers:
    # see Fodor, Fingscheidt - Reference-free SNR Measurement for Narrowband and Wideband Speech Signals in Car Noise
    if fs == 8e3
        m = 16.043
        s = 11.252
        p = [-2.1312, 6.4129, 2.0957, -19.199, 5.0992, 19.709, -7.7268, -6.6348, 2.0857, 13.066, 11.555]
    else
        m = 15.461
        s = 11.798
        p = [-.80823, 2.8537, -.3609, -7.5337, 4.3304, 7.2828, -5.0623, -1.8734, 0.88424, 13.06, 11.48]
    end

    snrvec = ((rawSNR - m)/ s) .^ collect(0:10)
    trueSNR = max(-11, p'snrvec)

    println("$(rawSNR), $(trueSNR)")

    return 0
end

end
