package nn

import "testing"

// TestAutoencoder_TrainingNormalizationStaysBounded confirms the invariant
// Normalize01 is supposed to provide during training: since MinVal/MaxVal
// expand to cover every value seen when updateBounds=true, norm[i] must
// always land in [0,1].
func TestAutoencoder_TrainingNormalizationStaysBounded(t *testing.T) {
	ae := NewAutoencoder(4, 0.75, 0.1)

	inputs := [][]float64{
		{0, 0, 0, 0},
		{10, -5, 1000, 0.5},
		{-100, 200, 1, 1},
		{5, 5, 5, 5},
	}

	for _, x := range inputs {
		ae.TrainStep(x)
		for i, v := range ae.scratchNorm {
			if v < 0 || v > 1 {
				t.Errorf("training-time normalized value out of [0,1]: feature %d = %v", i, v)
			}
		}
	}
}

// TestAutoencoder_PredictClampsOutOfRangeInput is a regression test for the
// bug where Normalize01 does not clamp its output during execution
// (updateBounds=false). Once MinVal/MaxVal are frozen after training, any
// feature value outside that frozen range must still normalize into [0,1]
// -- otherwise the reconstruction error (and therefore the anomaly score)
// is unbounded, which is what produced multi-order-of-magnitude score
// blowups (observed up to ~1e17) on real traffic.
//
// This test currently FAILS against the unpatched Normalize01/Predict and
// is expected to PASS once the clamp described in APPLY_FIXES.md is added.
func TestAutoencoder_PredictClampsOutOfRangeInput(t *testing.T) {
	ae := NewAutoencoder(4, 0.75, 0.1)

	// Train on a narrow but non-degenerate range (0.9-1.1) so MinVal/MaxVal
	// end up tight *and nonzero-width* -- a zero-width range would route
	// through Normalize01's "diff <= 1e-16" safe branch and never exercise
	// the buggy unclamped-division branch we're testing here.
	for i := 0; i < 50; i++ {
		v := 0.95
		if i%2 == 0 {
			v = 1.05
		}
		ae.TrainStep([]float64{1, 1, v, 1})
	}

	// A single feature spikes wildly beyond anything seen in training --
	// e.g. a burst statistic during an unusual flow.
	score := ae.Predict([]float64{1, 1, 1_000_000, 1})

	// The maximum possible per-feature squared error once normalization is
	// properly clamped to [0,1] is 1.0, so RMSE can never exceed 1.0.
	if score > 1.0001 {
		t.Errorf("expected Predict to return a bounded RMSE (<=1.0) even for wildly out-of-range input, got %v -- Normalize01 is not clamping its output to [0,1] during execution", score)
	}
}
