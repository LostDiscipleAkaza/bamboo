# How to apply the code fixes

These are hand-written patches (not a machine-generated diff) because the
GitHub web view strips original indentation from raw text — apply them with
`gofmt` afterward and you're safe either way.

## 0. `nn/autoencoder.go` — unbounded normalization causes exploding anomaly scores

**Confirmed root cause**, reproduced deterministically (identical config run
twice back-to-back produced bit-for-bit identical `ensemble_count`,
`threshold`, and `max_score`, ruling out a separate non-determinism bug).

`Normalize01` is supposed to scale every feature into `[0, 1]`:

```go
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
			norm[i] = (val - ae.MinVal[i]) / diff
		}
	}
	return norm
}
```

During training (`TrainStep`, `updateBounds=true`), `MinVal`/`MaxVal` expand
to cover every value seen, so `norm[i]` is mathematically guaranteed to land
in `[0,1]`. During execution (`Predict`, `updateBounds=false`), those bounds
are frozen at whatever they were when the grace period ended — so any
feature value that exceeds the frozen range produces a `norm[i]` **with no
ceiling or floor**. Meanwhile the reconstruction `z[i]` is always
sigmoid-bounded to `(0,1)`, so the squared error `(norm[i] - z[i])^2` can
become arbitrarily large. This compounds across the two-layer ensemble (the
L2 output autoencoder runs the same unclamped normalization again on the
L1 layer's already-inflated RMSE outputs), which is why observed `max_score`
values ranged from under 1.0 up to ~1.7e17 on the same traffic depending on
how far a flow's features fell outside the training window.

Replace the `else` branch with a clamp:

```go
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
```

This bounds any single autoencoder's max possible RMSE to `1.0`, which keeps
the log-normal threshold comparison meaningful instead of occasionally
comparing against 17-digit numbers. See `nn_autoencoder_test.go` for a
regression test — `TestAutoencoder_PredictClampsOutOfRangeInput` fails
against the unpatched code and passes once this clamp is applied.

## 1. `pipeline.go` — stop silently swallowing CSV write errors

**In `NewPipeline`**, replace:

```go
p.csvWriter.Write([]string{"packet", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "score", "threshold", "phase"})
p.csvWriter.Flush()
```

with:

```go
if err := p.csvWriter.Write([]string{"packet", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "score", "threshold", "phase"}); err != nil {
	return nil, fmt.Errorf("failed to write CSV header to %s: %w", cfg.CSVOutput, err)
}
p.csvWriter.Flush()
if err := p.csvWriter.Error(); err != nil {
	return nil, fmt.Errorf("failed to flush CSV header to %s: %w", cfg.CSVOutput, err)
}
```

**In `ProcessPacket`**, replace:

```go
if p.csvWriter != nil {
	phase := "exec"
	if isTraining {
		phase = "train"
	}
	p.csvWriter.Write([]string{
		strconv.Itoa(p.stats.TotalPackets),
		strconv.FormatFloat(meta.Timestamp, 'f', 6, 64),
		meta.SrcIP, meta.SrcPort, meta.DstIP, meta.DstPort, meta.Protocol,
		strconv.FormatFloat(score, 'f', 6, 64),
		strconv.FormatFloat(p.bambooSys.Threshold, 'f', 6, 64),
		phase,
	})

	if p.stats.TotalPackets%csvFlushInterval == 0 {
		p.csvWriter.Flush()
	}
}
```

with:

```go
if p.csvWriter != nil {
	phase := "exec"
	if isTraining {
		phase = "train"
	}
	if err := p.csvWriter.Write([]string{
		strconv.Itoa(p.stats.TotalPackets),
		strconv.FormatFloat(meta.Timestamp, 'f', 6, 64),
		meta.SrcIP, meta.SrcPort, meta.DstIP, meta.DstPort, meta.Protocol,
		strconv.FormatFloat(score, 'f', 6, 64),
		strconv.FormatFloat(p.bambooSys.Threshold, 'f', 6, 64),
		phase,
	}); err != nil {
		slog.Error("Failed to write CSV row", "packet", p.stats.TotalPackets, "error", err)
	}

	if p.stats.TotalPackets%csvFlushInterval == 0 {
		p.csvWriter.Flush()
		if err := p.csvWriter.Error(); err != nil {
			slog.Error("Failed to flush CSV buffer", "packet", p.stats.TotalPackets, "error", err)
		}
	}
}
```

No new imports are needed — `fmt` and `log/slog` are already imported in this file.

## 2. `pipeline.go` — sanitize `MinScore` when nothing has executed yet

`PipelineStats.MinScore` starts at `math.MaxFloat64` and is only updated once
a packet reaches execution phase. If a caller reads `Pipeline.Stats()` before
any packet exits the grace period (e.g. a very short capture, or a dashboard
polling early), they get back `1.7976931348623157e+308` instead of something
sane. Replace:

```go
// Stats returns a copy of current pipeline statistics
func (p *Pipeline) Stats() PipelineStats {
	if p.bambooSys != nil {
		p.stats.Threshold = p.bambooSys.Threshold
	}
	return p.stats
}
```

with:

```go
// Stats returns a copy of current pipeline statistics
func (p *Pipeline) Stats() PipelineStats {
	if p.bambooSys != nil {
		p.stats.Threshold = p.bambooSys.Threshold
	}
	stats := p.stats
	if stats.ExecPackets == 0 {
		stats.MinScore = 0
	}
	return stats
}
```

## 3. `README.md` — document the config keys that already exist in `config.go`

`config.go` defines `csv_output`, `decay_threshold`, `snap_len`,
`num_features`, `max_cluster_m`, and `metrics_port`, but the README's
configuration table only documents 12 of the 18 fields. Replace the table
with:

```markdown
| Parameter          | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `pcap_path`        | Path to a PCAP file. Leave empty for live capture.                   |
| `interface`        | Network interface for live capture (e.g. `eth0`).                    |
| `snap_len`         | Max bytes captured per packet (snaplen) for live capture.            |
| `bpf_filter`       | Optional BPF filter expression (e.g. `ip and not broadcast`).        |
| `csv_output`       | Path to write the anomaly-score CSV log to.                          |
| `csv_enabled`      | Toggle writing anomaly scores to CSV (`true`/`false`).                |
| `fm_grace_period`  | Packets used to learn feature clustering ($N\_{FM}$).                |
| `ad_grace_period`  | Packets used to train the autoencoder ensemble ($N\_{AD}$).          |
| `threshold_beta`   | Sigma multiplier for the log-normal anomaly threshold ($\beta$).     |
| `num_features`     | Dimensionality of the extracted feature vector (100 by default).    |
| `max_cluster_m`    | Max feature-cluster size used by the feature mapper.                 |
| `model_save_path`  | Path to save the trained model binary (e.g. `models/bamboo.bin`).    |
| `model_load_path`  | Path to load a pre-trained model (skips grace periods).              |
| `cleanup_interval` | Inactivity decay sweep interval in packets (prevents memory growth). |
| `decay_threshold`  | Decayed-weight cutoff below which an inactive entry is evicted.      |
| `test_mode`        | Human-readable terminal output instead of JSON.                      |
| `metrics_enabled`  | Expose a Prometheus metrics endpoint.                                |
| `metrics_port`     | Port the Prometheus metrics endpoint listens on.                     |
```
