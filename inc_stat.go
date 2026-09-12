// Package main implements incremental damped statistics for network traffic monitoring.
//
// Theoretical Foundation:
//   - Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS 2018 (Section 3.1: AfterImage).
//   - Welford, B. P., "Note on a Method for Calculating Corrected Sums of Squares and Products", Technometrics 1962.
package main

import (
	"math"
)

// IncStat maintains rolling damped 1D statistics (matches original AfterImage incStat)
type IncStat struct {
	Lambda      float64 // Decay factor λ
	Tlast       float64 // Last packet arrival time (seconds)
	W           float64 // Damped weight (packet count)
	LS          float64 // Linear sum of values (CF1)
	SS          float64 // Sum of squared values (CF2)
	Initialized bool
}

// creates a new incremental statistic for a given lambda
func NewIncStat(lambda float64) *IncStat {
	return &IncStat{
		Lambda:      lambda,
		Tlast:       0,
		W:           1e-20, // match original: avoid division by zero
		LS:          0,
		SS:          0,
		Initialized: false,
	}
}

// Update decays past state and inserts new observation x at timestamp t
func (s *IncStat) Update(x float64, t float64) {
	if s.Initialized {
		dt := t - s.Tlast
		if dt > 0 {
			gamma := math.Exp2(-s.Lambda * dt)
			s.W *= gamma
			s.LS *= gamma
			s.SS *= gamma
		}
	}
	s.Tlast = t
	s.Initialized = true

	s.LS += x
	s.SS += x * x
	s.W += 1.0
}

// returns the decayed packet count (rate proxy)
func (s *IncStat) Weight() float64 {
	return s.W
}

// DecayedWeight returns the estimated weight decayed to currentTime without mutating internal state
func (s *IncStat) DecayedWeight(currentTime float64) float64 {
	if !s.Initialized || currentTime <= s.Tlast {
		return s.W
	}
	dt := currentTime - s.Tlast
	return s.W * math.Exp2(-s.Lambda*dt)
}

// returns the running average
func (s *IncStat) Mean() float64 {
	if s.W <= 0 {
		return 0
	}
	return s.LS / s.W
}

// returns the running variance (matches original: abs(CF2/w - mean^2))
func (s *IncStat) Variance() float64 {
	if s.W <= 0 {
		return 0
	}
	mean := s.Mean()
	variance := math.Abs(s.SS/s.W - mean*mean)
	return variance
}

// returns the standard deviation
func (s *IncStat) StdDev() float64 {
	return math.Sqrt(s.Variance())
}

// AllStats1D returns [weight, mean, variance] matching original allstats_1D()
func (s *IncStat) AllStats1D() (float64, float64, float64) {
	mean := s.Mean()
	variance := math.Abs(s.SS/s.W - mean*mean)
	return s.W, mean, variance
}

// IncStatCov tracks the bidirectional covariance between two streams
// using the residual-product approach from the original AfterImage incStat_cov
type IncStatCov struct {
	Lambda float64

	// Per-stream 1D stats (embedded — each side of the bidirectional flow)
	Streams [2]IncStat

	// Covariance tracking
	CovTlast float64    // last timestamp for covariance decay
	CovW     float64    // covariance weight
	CF3      float64    // sum of cross-residual products
	LastRes  [2]float64 // last residual per stream
}

func NewIncStatCov(lambda float64) *IncStatCov {
	return &IncStatCov{
		Lambda: lambda,
		Streams: [2]IncStat{
			{Lambda: lambda, W: 1e-20},
			{Lambda: lambda, W: 1e-20},
		},
		CovW: 1e-20,
	}
}

// Update processes a new observation for the given stream (0 or 1).
// This matches the original's combined update_get_1D_Stats + update_cov flow.
func (c *IncStatCov) Update(streamIdx int, v float64, t float64) {
	otherIdx := 1 - streamIdx

	// 1. Update the current stream's 1D stats
	c.Streams[streamIdx].Update(v, t)

	// 2. Decay the OTHER stream's 1D stats to current time (without adding a value)
	c.decayStream(otherIdx, t)

	// 3. Decay covariance state
	c.decayCov(t, streamIdx)

	// 4. Compute current stream's residual (using updated mean)
	res := v - c.Streams[streamIdx].Mean()

	// 5. Cross-product with other stream's last residual
	crossResid := res * c.LastRes[otherIdx]
	c.CF3 += crossResid
	c.CovW += 1.0

	// 6. Store current residual
	c.LastRes[streamIdx] = res
}

// decayStream decays a stream's 1D stats to timestamp t without adding a value
func (c *IncStatCov) decayStream(idx int, t float64) {
	s := &c.Streams[idx]
	if s.Initialized {
		dt := t - s.Tlast
		if dt > 0 {
			gamma := math.Exp2(-s.Lambda * dt)
			s.W *= gamma
			s.LS *= gamma
			s.SS *= gamma
			s.Tlast = t
		}
	}
}

// decayCov decays the covariance state
func (c *IncStatCov) decayCov(t float64, streamIdx int) {
	dt := t - c.CovTlast
	if dt > 0 {
		gamma := math.Exp2(-c.Lambda * dt)
		c.CF3 *= gamma
		c.CovW *= gamma
		c.CovTlast = t
		c.LastRes[streamIdx] *= gamma
	}
}

// IsDecayed returns true if all stream weights and covariance weight have decayed below threshold
func (c *IncStatCov) IsDecayed(currentTime float64, threshold float64) bool {
	w0 := c.Streams[0].DecayedWeight(currentTime)
	w1 := c.Streams[1].DecayedWeight(currentTime)
	covW := c.CovW
	if currentTime > c.CovTlast {
		covW *= math.Exp2(-c.Lambda * (currentTime - c.CovTlast))
	}
	return w0 < threshold && w1 < threshold && covW < threshold
}

// Covariance returns the covariance estimate
func (c *IncStatCov) Covariance() float64 {
	if c.CovW <= 0 {
		return 0
	}
	return c.CF3 / c.CovW
}

// Correlation returns the Pearson correlation coefficient
func (c *IncStatCov) Correlation() float64 {
	cov := c.Covariance()
	ss := c.Streams[0].StdDev() * c.Streams[1].StdDev()
	if ss == 0 {
		return 0
	}
	return cov / ss
}

// Radius returns sqrt(var_src^2 + var_dst^2) — L2 norm of variances
func (c *IncStatCov) Radius() float64 {
	v0 := c.Streams[0].Variance()
	v1 := c.Streams[1].Variance()
	return math.Sqrt(v0*v0 + v1*v1)
}

// Magnitude returns sqrt(mean_src^2 + mean_dst^2) — L2 norm of means
func (c *IncStatCov) Magnitude() float64 {
	m0 := c.Streams[0].Mean()
	m1 := c.Streams[1].Mean()
	return math.Sqrt(m0*m0 + m1*m1)
}

// Stats2D returns [radius, magnitude, covariance, pcc] matching original get_stats2()
func (c *IncStatCov) Stats2D() (float64, float64, float64, float64) {
	return c.Radius(), c.Magnitude(), c.Covariance(), c.Correlation()
}
