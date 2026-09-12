# Bamboo NIDS — Findings & Fixes

This document summarizes everything found while testing Bamboo against a
real packet capture, the root cause of each issue, and the fix included in
this PR. Issues are ordered by severity.

---

## 1. Unbounded feature normalization causes exploding anomaly scores (Critical)

**File:** `nn/autoencoder.go` — `Normalize01`

**Symptom:** On a benign ~7,000-packet LAN capture (mostly mDNS/SSDP/DHCP
background traffic), `max_score` varied across test runs from under 1.0 up
to `1.7e17`, against a log-normal threshold that's normally in the 1–3
range. A single QUIC (UDP/443) flow not seen during the training window
scored 1,743,034 — roughly 700,000x the threshold.

**Root cause:** `Normalize01` is meant to scale every feature into `[0, 1]`:

```go
diff := ae.MaxVal[i] - ae.MinVal[i]
if diff <= 1e-16 {
    norm[i] = 0.0
} else {
    norm[i] = (val - ae.MinVal[i]) / diff
}
```

During training (`TrainStep`, `updateBounds=true`), `MinVal`/`MaxVal` expand
to cover every value seen, so `norm[i]` is mathematically guaranteed to fall
in `[0,1]`. During execution (`Predict`, `updateBounds=false`), those
bounds are frozen at whatever they were when the grace period ended. Any
feature value that exceeds that frozen range produces a `norm[i]` with
**no ceiling or floor**. The reconstruction `z[i]` is always
sigmoid-bounded to `(0,1)`, so the squared error `(norm[i] - z[i])^2` can
become arbitrarily large — and this compounds, since the L2 output
autoencoder runs the same unclamped normalization again on the L1 layer's
already-inflated RMSE outputs.

**Reproduced deterministically:** running the identical config twice in a
row against the same pcap produced bit-for-bit identical
`ensemble_count`/`threshold`/`max_score`, confirming this is a real
algorithmic bug and not run-to-run non-determinism.

**Fix:** clamp the normalized value to `[0, 1]`:

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

This bounds any single autoencoder's max possible RMSE to `1.0`, keeping
the log-normal threshold comparison meaningful.

**Test added:** `nn/autoencoder_test.go` —
`TestAutoencoder_PredictClampsOutOfRangeInput` fails against the unpatched
code and passes once the clamp is applied.
`TestAutoencoder_TrainingNormalizationStaysBounded` documents the invariant
that already holds correctly during training.

---

## 2. Live capture ignores Ctrl+C while idle (High)

**Files:** `main.go`, `pipeline.go`

**Symptom:** Running live capture (`sudo ./bamboo-nids -c config.yaml`
with `interface: eth0`) on an idle interface does not respond to Ctrl+C at
all.

**Root cause:** `Pipeline.Run` correctly selects on both `ctx.Done()` and
the packet channel:

```go
select {
case <-ctx.Done():
    return ctx.Err()
case packet, ok := <-packetChan:
    ...
}
```

But `packetSource.Packets()` is fed by a blocking libpcap C call
(`pcap_next_ex` via cgo, opened with `pcap.BlockForever`) that only returns
when a packet arrives. Go's context cancellation cannot interrupt a
blocked cgo call. `main.go`'s signal handler only calls `cancel()` — it
never closes the pcap handle, so on an idle interface the process hangs
until traffic arrives or it's killed with `SIGKILL`.

**Fix:** close the handle from the signal-handling goroutine so the
blocked read is interrupted immediately:

```go
go func() {
    sig := <-sigChan
    slog.Warn("Received shutdown signal", "signal", sig.String())
    cancel()
    handle.Close() // unblocks pcap_next_ex on an idle interface
}()
```

(`initCapture` needs to run before this goroutine is set up so `handle` is
in scope; see the diff in this PR.)

---

## 3. CSV write/flush errors are silently discarded (Low)

**File:** `pipeline.go` — `NewPipeline`, `ProcessPacket`

**Symptom:** If the CSV output can't be written (disk full, permissions,
etc.), `bamboo-nids` doesn't tell you — it just quietly stops producing a
usable results file.

**Fix:** check and log/return the error from every `csvWriter.Write(...)`
and `csvWriter.Flush()`/`.Error()` call. See the diff in this PR for the
exact before/after for both the header write in `NewPipeline` and the
per-packet write in `ProcessPacket`.

---

## 4. `Stats()` can return `math.MaxFloat64` as `MinScore` (Low)

**File:** `pipeline.go` — `PipelineStats`, `Pipeline.Stats()`

**Symptom:** `PipelineStats.MinScore` is initialized to `math.MaxFloat64`
and only updated once a packet reaches execution phase. Any caller of
`Pipeline.Stats()` before that point (e.g. a very short capture, or a
dashboard polling early) gets back `1.7976931348623157e+308` instead of
something sane.

**Fix:** sanitize `MinScore` to `0` in `Stats()` when `ExecPackets == 0`.

---

## 5. README configuration table is missing 6 documented fields (Doc)

**File:** `README.md`

`config.go` defines `csv_output`, `decay_threshold`, `snap_len`,
`num_features`, `max_cluster_m`, and `metrics_port`, but the README's
config table only documented 12 of the 18 fields. Updated table included
in this PR.

---

## Testing & CI added

No test files existed in the repo prior to this PR, despite the codebase
implementing specific numerical algorithms (Welford's online variance,
exponential decay, correntropy-based clustering, tied-weight autoencoders)
where a silent regression is easy to introduce and hard to catch by eye.

| File | Covers |
|---|---|
| `inc_stat_test.go` | Damped incremental stats: decay behavior, mean/variance on a constant stream, covariance/correlation sign and bounds, `IsDecayed` eviction logic |
| `net_stat_test.go` | Canonical host/socket pair key ordering (order-independence), 100-feature output length, ARP handling, `Cleanup` eviction |
| `nn/autoencoder_test.go` | The normalization-bounds invariant during training, and the regression test for Finding #1 |
| `.github/workflows/ci.yml` | `go build`, `go vet`, `go test -race`, and `gofmt -l` on every push/PR |
| `CONTRIBUTING.md` | Setup instructions and a pre-PR checklist (format, vet, test, keep README/config in sync) |

---

## How to verify this PR

```
go build ./...
go vet ./...
go test ./... -v
```

To see Finding #1 concretely: run `./bamboo-nids -c config.yaml` against
any pcap before and after the `Normalize01` fix. Before: `max_score` can
reach into the millions or higher on ordinary traffic. After: `max_score`
should stay in a range comparable to the computed `threshold` (single or
low double digits for most traffic), since no per-feature error can exceed
`1.0` anymore.
