package main

import (
	"math"
	"testing"
)

const epsilon = 1e-6

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestNewIncStat(t *testing.T) {
	s := NewIncStat(1.0)
	if s.Lambda != 1.0 {
		t.Errorf("expected Lambda 1.0, got %f", s.Lambda)
	}
	if s.W != 1e-20 {
		t.Errorf("expected initial weight 1e-20, got %v", s.W)
	}
	if s.Mean() != 0 {
		t.Errorf("expected initial mean 0, got %v", s.Mean())
	}
	if s.Variance() != 0 {
		t.Errorf("expected initial variance 0, got %v", s.Variance())
	}
}

func TestIncStat_FirstUpdateNoDecay(t *testing.T) {
	// The very first Update call must not apply decay (Tlast starts at 0),
	// otherwise the initial epsilon weight would be scaled for no reason.
	s := NewIncStat(5.0)
	s.Update(10.0, 0.0)

	if !almostEqual(s.LS, 10.0, epsilon) {
		t.Errorf("expected LS=10.0 after first update, got %v", s.LS)
	}
	if !almostEqual(s.SS, 100.0, epsilon) {
		t.Errorf("expected SS=100.0 after first update, got %v", s.SS)
	}
	if !almostEqual(s.Mean(), 10.0, 1e-10) {
		t.Errorf("expected mean ~10.0, got %v", s.Mean())
	}
}

func TestIncStat_DecayReducesWeightOverTime(t *testing.T) {
	s := NewIncStat(1.0)
	// Note: the very first Update never decays regardless of the timestamp
	// passed (Tlast starts at 0), so we need a *second* update at a later,
	// nonzero timestamp to actually exercise the decay path.
	s.Update(10.0, 1.0)
	wAfterFirst := s.Weight()

	s.Update(10.0, 6.0) // 5 seconds later, same value
	// Weight should have decayed then had 1.0 added back; verify the decayed
	// portion is meaningfully smaller than a full undamped weight of +1.0.
	if s.Weight() >= wAfterFirst+1.0 {
		t.Errorf("expected decay to reduce carried-over weight, got %v (was %v)", s.Weight(), wAfterFirst)
	}

	// A long gap should decay the weight close to (but not below) the new
	// observation's contribution.
	decayed := s.DecayedWeight(10000.0)
	if decayed > 1.01 {
		t.Errorf("expected weight to have decayed close to the epsilon floor after a long gap, got %v", decayed)
	}
}

func TestIncStat_DecayedWeightDoesNotMutateState(t *testing.T) {
	s := NewIncStat(2.0)
	s.Update(1.0, 0.0)
	wField := s.W
	tField := s.Tlast

	_ = s.DecayedWeight(500.0)

	if s.W != wField || s.Tlast != tField {
		t.Errorf("DecayedWeight must not mutate internal state, W changed %v->%v, Tlast changed %v->%v", wField, s.W, tField, s.Tlast)
	}
}

func TestIncStat_MeanAndVarianceConstantStream(t *testing.T) {
	s := NewIncStat(0.01) // slow decay so weight stays high
	for i := 0; i < 50; i++ {
		s.Update(7.0, 1.0+float64(i)*0.001)
	}
	if !almostEqual(s.Mean(), 7.0, 1e-3) {
		t.Errorf("expected mean ~7.0 for a constant stream, got %v", s.Mean())
	}
	if s.Variance() > 1e-3 {
		t.Errorf("expected ~0 variance for a constant stream, got %v", s.Variance())
	}
}

func TestIncStatCov_PositiveCorrelationForIdenticalStreams(t *testing.T) {
	c := NewIncStatCov(0.01)
	// Feed matching values into both streams, alternating, so residuals track together.
	for i := 0; i < 30; i++ {
		v := float64(10 + i%5)
		tm := float64(i) * 0.01
		c.Update(0, v, tm)
		c.Update(1, v, tm+0.001)
	}
	corr := c.Correlation()
	if corr <= 0 {
		t.Errorf("expected positive correlation for two streams fed the same values, got %v", corr)
	}
	if corr > 1.0001 {
		t.Errorf("correlation must not exceed 1.0, got %v", corr)
	}
}

func TestIncStatCov_IsDecayed(t *testing.T) {
	c := NewIncStatCov(2.0)
	c.Update(0, 5.0, 0.0)
	c.Update(1, 5.0, 0.0)

	if c.IsDecayed(0.0, 0.001) {
		t.Errorf("freshly updated stats should not be considered decayed")
	}
	if !c.IsDecayed(10000.0, 0.001) {
		t.Errorf("stats untouched for a long time should be considered decayed")
	}
}

func TestIncStatCov_Stats2DShape(t *testing.T) {
	c := NewIncStatCov(1.0)
	c.Update(0, 3.0, 0.0)
	c.Update(1, 4.0, 0.0)

	radius, magnitude, _, pcc := c.Stats2D()
	if radius < 0 {
		t.Errorf("radius must be non-negative, got %v", radius)
	}
	if magnitude < 0 {
		t.Errorf("magnitude must be non-negative, got %v", magnitude)
	}
	if pcc < -1.0001 || pcc > 1.0001 {
		t.Errorf("pearson correlation coefficient must be in [-1, 1], got %v", pcc)
	}
}
