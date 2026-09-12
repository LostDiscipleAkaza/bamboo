package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/google/gopacket"
	"github.com/lsrik19/bamboo-nids/nn"
)

const (
	defaultFMProgressInterval = 10000
	defaultADProgressInterval = 50000
	defaultExecStatusInterval = 50000
	csvFlushInterval          = 10000
)

// PipelineStats holds running counters and performance metrics of the pipeline
type PipelineStats struct {
	TotalPackets      int
	ExecPackets       int
	AnomaliesDetected int
	MaxScore          float64
	MinScore          float64
	ScoreSum          float64
	Threshold         float64
}

// Pipeline coordinates packet capture, feature extraction, neural ensemble inference, and output logging
type Pipeline struct {
	cfg           *Config
	netStat       *NetStat
	featureMapper *nn.FeatureMapper
	bambooSys     *nn.Bamboo
	csvWriter     *csv.Writer
	csvFile       *os.File
	modelLoaded   bool
	modelSaved    bool
	stats         PipelineStats
}

// NewPipeline initializes all stages of the NIDS pipeline
func NewPipeline(cfg *Config) (*Pipeline, error) {
	p := &Pipeline{
		cfg:     cfg,
		netStat: NewNetStat(),
		stats: PipelineStats{
			MinScore: math.MaxFloat64,
		},
	}

	// Initialize or load the neural network model
	if cfg.ModelLoadPath != "" {
		slog.Info("Loading pre-trained model", "path", cfg.ModelLoadPath)
		loadedModel, err := nn.LoadModel(cfg.ModelLoadPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load pre-trained model %s: %w", cfg.ModelLoadPath, err)
		}
		p.featureMapper = loadedModel.FeatureMapper
		p.bambooSys = loadedModel.Bamboo
		p.modelLoaded = true
		slog.Info("Pre-trained model loaded successfully",
			"created_at", loadedModel.CreatedAt.Format(time.RFC3339),
			"num_features", loadedModel.NumFeatures,
			"clusters", len(p.featureMapper.Clusters),
			"threshold", p.bambooSys.Threshold,
		)
	} else {
		p.featureMapper = nn.NewFeatureMapper(cfg.NumFeatures, cfg.MaxClusterM, cfg.FMGracePeriod)
		p.bambooSys = nn.NewBamboo(cfg.ADGracePeriod, cfg.ThresholdBeta)
	}

	// Initialize CSV writer if enabled
	if cfg.IsCSVEnabled() {
		if err := os.MkdirAll(filepath.Dir(cfg.CSVOutput), 0755); err != nil {
			return nil, fmt.Errorf("failed to create results directory: %w", err)
		}
		f, err := os.Create(cfg.CSVOutput)
		if err != nil {
			return nil, fmt.Errorf("failed to create CSV file %s: %w", cfg.CSVOutput, err)
		}
		p.csvFile = f
		p.csvWriter = csv.NewWriter(f)
		if err := p.csvWriter.Write([]string{"packet", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "score", "threshold", "phase"}); err != nil {
			return nil, fmt.Errorf("failed to write CSV header to %s: %w", cfg.CSVOutput, err)
		}
		p.csvWriter.Flush()
		if err := p.csvWriter.Error(); err != nil {
			return nil, fmt.Errorf("failed to flush CSV header to %s: %w", cfg.CSVOutput, err)
		}
	}

	return p, nil
}

// Run processes incoming packets until the context is cancelled or the stream ends
func (p *Pipeline) Run(ctx context.Context, packetSource *gopacket.PacketSource) error {
	slog.Info("Pipeline started. Processing packets...")
	packetChan := packetSource.Packets()

	for {
		select {
		case <-ctx.Done():
			slog.Info("Context cancelled, stopping pipeline.")
			return ctx.Err()
		case packet, ok := <-packetChan:
			if !ok {
				slog.Info("Reached end of packet stream.")
				return nil
			}
			p.ProcessPacket(packet)
		}
	}
}

// ProcessPacket executes feature extraction, mapping, and anomaly scoring for a single packet
func (p *Pipeline) ProcessPacket(packet gopacket.Packet) {
	p.stats.TotalPackets++
	metricPacketsProcessed.Inc()

	meta := parsePacket(packet)
	if meta == nil {
		return
	}

	// Extract damped incremental statistics
	x := p.netStat.UpdateAndExtract(meta)

	// Periodic decay-based memory eviction
	if p.cfg.CleanupInterval > 0 && p.stats.TotalPackets%p.cfg.CleanupInterval == 0 {
		threshold := p.cfg.DecayThreshold
		if threshold <= 0 {
			threshold = 0.001
		}
		evicted := p.netStat.Cleanup(meta.Timestamp, threshold)
		if evicted > 0 {
			slog.Debug("Memory cleanup completed",
				"evicted_entries", evicted,
				"remaining_entries", p.netStat.TotalEntries(),
				"packet", p.stats.TotalPackets,
			)
		}
	}

	// Feature mapping
	v, isFMReady := p.featureMapper.Process(x)
	if !isFMReady {
		if p.stats.TotalPackets%defaultFMProgressInterval == 0 {
			slog.Debug("FM Grace Period", "packet", p.stats.TotalPackets, "required", p.cfg.FMGracePeriod)
		}
		return
	}

	// Initialize ensemble on first transition out of FM grace period
	if p.bambooSys.EnsembleLayer == nil {
		p.bambooSys.InitEnsemble(p.featureMapper.Clusters)
		slog.Info("Feature Mapper ready. Autoencoder ensemble initialized.")
	}

	// Anomaly detection forward pass
	score, isAlert, isTraining := p.bambooSys.Process(v)

	// Save trained model immediately once grace period completes
	if p.cfg.ModelSavePath != "" && p.bambooSys.IsTrained && !p.modelSaved && !p.modelLoaded {
		if err := nn.SaveModel(p.cfg.ModelSavePath, p.featureMapper, p.bambooSys); err != nil {
			slog.Error("Failed to save trained model", "path", p.cfg.ModelSavePath, "error", err)
		} else {
			p.modelSaved = true
			slog.Info("Trained model saved successfully", "path", p.cfg.ModelSavePath, "threshold", p.bambooSys.Threshold)
		}
	}

	// Update Prometheus metrics
	metricCurrentScore.Set(score)
	metricThreshold.Set(p.bambooSys.Threshold)

	// Log to CSV if configured
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

	if isTraining {
		if p.stats.TotalPackets%defaultADProgressInterval == 0 {
			slog.Debug("AD Grace Period", "packet", p.bambooSys.Count, "required", p.cfg.ADGracePeriod, "rmse", score)
		}
	} else {
		// Execution mode
		p.stats.ExecPackets++
		p.stats.ScoreSum += score
		if score > p.stats.MaxScore {
			p.stats.MaxScore = score
		}
		if score < p.stats.MinScore {
			p.stats.MinScore = score
		}

		if isAlert {
			p.stats.AnomaliesDetected++
			metricAlertsGenerated.Inc()

			// Only log every single packet anomaly if NOT in TestMode
			if !p.cfg.TestMode {
				slog.Warn("Anomaly Detected",
					"score", score,
					"threshold", p.bambooSys.Threshold,
					"src_ip", meta.SrcIP,
					"dst_ip", meta.DstIP,
					"protocol", meta.Protocol,
					"packet_id", p.stats.TotalPackets,
				)
			}
		}

		if p.stats.ExecPackets%defaultExecStatusInterval == 0 {
			slog.Info("Execution Status",
				"packets_scored", p.stats.ExecPackets,
				"alerts", p.stats.AnomaliesDetected,
				"max_score", p.stats.MaxScore,
				"avg_score", p.stats.ScoreSum/float64(p.stats.ExecPackets),
			)
		}
	}
}

// Close flushes CSV buffers, saves model if trained but unsaved, and closes resources
func (p *Pipeline) Close() error {
	if p.csvWriter != nil {
		p.csvWriter.Flush()
	}
	if p.csvFile != nil {
		p.csvFile.Close()
	}

	// Save model at completion if not already saved during the loop
	if p.cfg.ModelSavePath != "" && p.bambooSys != nil && p.bambooSys.IsTrained && !p.modelSaved && !p.modelLoaded {
		if err := nn.SaveModel(p.cfg.ModelSavePath, p.featureMapper, p.bambooSys); err != nil {
			slog.Error("Failed to save trained model at shutdown", "path", p.cfg.ModelSavePath, "error", err)
			return err
		}
		p.modelSaved = true
		slog.Info("Trained model saved successfully", "path", p.cfg.ModelSavePath, "threshold", p.bambooSys.Threshold)
	}

	return nil
}

// PrintSummary outputs final pipeline statistics
func (p *Pipeline) PrintSummary() {
	if p.bambooSys != nil {
		p.stats.Threshold = p.bambooSys.Threshold
	}
	slog.Info("Pipeline Summary",
		"total_packets", p.stats.TotalPackets,
		"exec_packets", p.stats.ExecPackets,
		"anomalies_detected", p.stats.AnomaliesDetected,
		"threshold", p.stats.Threshold,
		"max_score", p.stats.MaxScore,
	)
}

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
