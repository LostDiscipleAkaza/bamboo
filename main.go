package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/google/gopacket"
	"github.com/google/gopacket/pcap"
)

func initLogger(cfg *Config) {
	var logLevel slog.Level
	if err := logLevel.UnmarshalText([]byte(cfg.LogLevel)); err != nil {
		logLevel = slog.LevelInfo
	}

	var handler slog.Handler
	if cfg.TestMode {
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel})
	} else {
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel})
	}

	logger := slog.New(handler)
	slog.SetDefault(logger)
}

func initCapture(cfg *Config) (*pcap.Handle, error) {
	var handle *pcap.Handle
	var err error

	if cfg.PcapPath != "" {
		slog.Info("Opening offline PCAP", "path", cfg.PcapPath)
		handle, err = pcap.OpenOffline(cfg.PcapPath)
	} else if cfg.Interface != "" {
		slog.Info("Opening live interface", "interface", cfg.Interface)
		handle, err = pcap.OpenLive(cfg.Interface, cfg.SnapLen, true, pcap.BlockForever)
	} else {
		return nil, fmt.Errorf("configuration must specify either pcap_path or interface")
	}
	if err != nil {
		return nil, err
	}

	if cfg.BPFFilter != "" {
		if err := handle.SetBPFFilter(cfg.BPFFilter); err != nil {
			handle.Close()
			return nil, fmt.Errorf("failed to set BPF filter %q: %w", cfg.BPFFilter, err)
		}
		slog.Info("Applied BPF filter", "filter", cfg.BPFFilter)
	}

	return handle, nil
}

func main() {
	configPath := flag.String("c", "config.yaml", "Path to YAML configuration file")
	saveModelFlag := flag.String("save-model", "", "Path to save trained model (overrides config)")
	loadModelFlag := flag.String("load-model", "", "Path to load pre-trained model (overrides config)")
	bpfFlag := flag.String("bpf", "", "BPF filter expression (overrides config)")
	flag.Parse()

	// Load Configuration
	cfg, err := LoadConfig(*configPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	if *saveModelFlag != "" {
		cfg.ModelSavePath = *saveModelFlag
	}
	if *loadModelFlag != "" {
		cfg.ModelLoadPath = *loadModelFlag
	}
	if *bpfFlag != "" {
		cfg.BPFFilter = *bpfFlag
	}

	// Initialize Structured Logging
	initLogger(cfg)
	slog.Info("Starting Bamboo NIDS", "config", *configPath)

	// Setup Graceful Shutdown Context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize Packet Capture
	handle, err := initCapture(cfg)
	if err != nil {
		slog.Error("Capture initialization failed", "error", err)
		os.Exit(1)
	}
	defer handle.Close()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-sigChan
		slog.Warn("Received shutdown signal", "signal", sig.String())
		cancel()
		// pcap_next_ex blocks in a cgo call on a live, idle interface and
		// won't notice ctx.Done() until a packet arrives. Closing the handle
		// here interrupts that blocking read immediately so shutdown is
		// actually responsive.
		handle.Close()
	}()

	// Initialize Prometheus Metrics Server
	if cfg.MetricsEnabled {
		metricsSrv := startMetricsServer(cfg.MetricsPort)
		defer stopMetricsServer(metricsSrv)
	}

	// Initialize NIDS Processing Pipeline
	pipeline, err := NewPipeline(cfg)
	if err != nil {
		slog.Error("Pipeline initialization failed", "error", err)
		os.Exit(1)
	}
	defer pipeline.Close()

	// Run Pipeline
	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())
	_ = pipeline.Run(ctx, packetSource)

	// Print Final Summary
	pipeline.PrintSummary()
}
