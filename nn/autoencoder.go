// Theoretical Foundation:
//   - Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders", ICML 2008.
//   - Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS 2018 (Section 3.3).
package nn

import (
	"math"
	"math/rand"
)

// Autoencoder represents a single-hidden-layer denoising autoencoder with tied weights (W' = W.T)
type Autoencoder struct {
	NIn     int     // Visible input dimension
	NHidden int     // Bottleneck hidden layer dimension
	LR      float64 // Learning rate (typically 0.1)

	// Tied weights: W is (NIn x NHidden), decoder uses W transposed
	W     [][]float64 // (NIn x NHidden) — shared encoder/decoder weights
	HBias []float64   // (NHidden) — hidden layer bias
	VBias []float64   // (NIn) — visible layer (output) bias

	// Online 0-1 Normalizer bounds
	MinVal []float64
	MaxVal []float64
	N      int // number of training samples seen

	// Pre-allocated scratch buffers (zero-allocation hot path)
	scratchNorm []float64 // len = NIn
	scratchA1   []float64 // len = NHidden
	scratchY    []float64 // len = NIn
	scratchLh2  []float64 // len = NIn
	scratchLh1  []float64 // len = NHidden
}

func sigmoid(x float64) float64 {
	if x < -45.0 {
		return 0.0
	} else if x > 45.0 {
		return 1.0
	}
	return 1.0 / (1.0 + math.Exp(-x))
}

// NewAutoencoder initializes weights uniformly in [-1/n, 1/n] with default seed 1234 matching dA.py
func NewAutoencoder(nIn int, hiddenRatio float64, lr float64) *Autoencoder {
	return NewAutoencoderWithSeed(nIn, hiddenRatio, lr, 1234)
}

// NewAutoencoderWithSeed initializes weights uniformly with a custom RNG seed
func NewAutoencoderWithSeed(nIn int, hiddenRatio float64, lr float64, seed int64) *Autoencoder {
	nHidden := int(math.Ceil(float64(nIn) * hiddenRatio))
	if nHidden < 1 {
		nHidden = 1
	}

	// Original uses 1/n_visible (not 1/sqrt(n))
	limit := 1.0 / float64(nIn)

	rng := rand.New(rand.NewSource(seed))

	w := make([][]float64, nIn)
	for i := range w {
		w[i] = make([]float64, nHidden)
		for j := range w[i] {
			w[i][j] = (rng.Float64()*2.0 - 1.0) * limit
		}
	}

	minVal := make([]float64, nIn)
	maxVal := make([]float64, nIn)
	for i := 0; i < nIn; i++ {
		minVal[i] = math.Inf(1)
		maxVal[i] = math.Inf(-1)
	}

	ae := &Autoencoder{
		NIn:     nIn,
		NHidden: nHidden,
		LR:      lr,
		W:       w,
		HBias:   make([]float64, nHidden),
		VBias:   make([]float64, nIn),
		MinVal:  minVal,
		MaxVal:  maxVal,
		N:       0,
	}
	ae.InitScratchBuffers()
	return ae
}

// InitScratchBuffers allocates or resets transient scratch buffers (called after deserialization)
func (ae *Autoencoder) InitScratchBuffers() {
	ae.scratchNorm = make([]float64, ae.NIn)
	ae.scratchA1 = make([]float64, ae.NHidden)
	ae.scratchY = make([]float64, ae.NIn)
	ae.scratchLh2 = make([]float64, ae.NIn)
	ae.scratchLh1 = make([]float64, ae.NHidden)
}

// Normalize01 updates min/max bounds and scales x into [0, 1]
// Uses pre-allocated scratchNorm buffer to avoid per-call allocation.
func (ae *Autoencoder) Normalize01(x []float64, updateBounds bool) []float64 {
	norm := ae.scratchNorm
	for i, val := range x {
		if updateBounds {
			if val < ae.MinVal[i] {
				ae.MinVal[i] = val
			}
			if val > ae.MaxVal[i] {
				ae.MaxVal[i] = val
			}
		}

		diff := ae.MaxVal[i] - ae.MinVal[i]
		if diff <= 1e-16 {
			norm[i] = 0.0
		} else {
			v := (val - ae.MinVal[i]) / diff
			if v < 0 {
				v = 0
			} else if v > 1 {
				v = 1
			}
			norm[i] = v
		}
	}
	return norm
}

// Forward computes activations using tied weights
// Encode: a1 = sigmoid(x @ W + hbias)
// Decode: y  = sigmoid(a1 @ W.T + vbias) = sigmoid(sum_h(a1[h] * W[i][h]) + vbias[i])
func (ae *Autoencoder) Forward(xNorm []float64) ([]float64, []float64) {
	// Encode: a1[h] = sigmoid(sum_i(xNorm[i] * W[i][h]) + HBias[h])
	a1 := ae.scratchA1
	for h := 0; h < ae.NHidden; h++ {
		sum := ae.HBias[h]
		for i := 0; i < ae.NIn; i++ {
			sum += xNorm[i] * ae.W[i][h]
		}
		a1[h] = sigmoid(sum)
	}

	// Decode: y[i] = sigmoid(sum_h(a1[h] * W[i][h]) + VBias[i])
	// W.T[h][i] = W[i][h], so dot(a1, W.T) for output i = sum_h(a1[h] * W[i][h])
	y := ae.scratchY
	for i := 0; i < ae.NIn; i++ {
		sum := ae.VBias[i]
		for h := 0; h < ae.NHidden; h++ {
			sum += a1[h] * ae.W[i][h]
		}
		y[i] = sigmoid(sum)
	}

	return a1, y
}

// TrainStep performs a single-sample online SGD update with tied weights
// Matches original dA.py train() method exactly
func (ae *Autoencoder) TrainStep(x []float64) float64 {
	ae.N++
	xNorm := ae.Normalize01(x, true)
	a1, z := ae.Forward(xNorm)

	// L_h2 = x - z (reconstruction error, sign convention: target - output)
	Lh2 := ae.scratchLh2
	sse := 0.0
	for i := 0; i < ae.NIn; i++ {
		Lh2[i] = xNorm[i] - z[i]
		sse += Lh2[i] * Lh2[i]
	}
	rmse := math.Sqrt(sse / float64(ae.NIn))

	// L_h1 = dot(L_h2, W) * y * (1-y)   — hidden layer error
	Lh1 := ae.scratchLh1
	for h := 0; h < ae.NHidden; h++ {
		sum := 0.0
		for i := 0; i < ae.NIn; i++ {
			sum += Lh2[i] * ae.W[i][h]
		}
		Lh1[h] = sum * a1[h] * (1.0 - a1[h])
	}

	// L_W = outer(x, L_h1) + outer(L_h2, y)  — tied weight gradient
	// W += lr * L_W
	for i := 0; i < ae.NIn; i++ {
		for h := 0; h < ae.NHidden; h++ {
			lw := xNorm[i]*Lh1[h] + Lh2[i]*a1[h]
			ae.W[i][h] += ae.LR * lw
		}
	}

	// HBias += lr * L_h1
	for h := 0; h < ae.NHidden; h++ {
		ae.HBias[h] += ae.LR * Lh1[h]
	}

	// VBias += lr * L_h2
	for i := 0; i < ae.NIn; i++ {
		ae.VBias[i] += ae.LR * Lh2[i]
	}

	return rmse
}

// Predict calculates reconstruction RMSE without altering weights
func (ae *Autoencoder) Predict(x []float64) float64 {
	xNorm := ae.Normalize01(x, false)
	_, z := ae.Forward(xNorm)

	sse := 0.0
	for i := 0; i < ae.NIn; i++ {
		diff := xNorm[i] - z[i]
		sse += diff * diff
	}
	return math.Sqrt(sse / float64(ae.NIn))
}
