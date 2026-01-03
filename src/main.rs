use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod evaluation;
mod metrics;
mod preprocessing;
mod resolve;
mod geometry;
mod model;

use evaluation::Evaluator;
use preprocessing::NormalizationForm;

/// Evaluate Mass Digitalization Data
#[derive(Parser, Debug)]
#[command(name = "digital-eval")]
#[command(version = "1.9.1")]
#[command(about = "Evaluate Mass Digitalization Data", long_about = None)]
struct Args {
    /// Root Directory for evaluation candidates
    #[arg(value_name = "CANDIDATES")]
    candidates: PathBuf,

    /// Root directory for Reference/Groundtruth data
    #[arg(short = 'r', long = "reference")]
    reference: Option<PathBuf>,

    /// Verbosity level (can be used multiple times: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbosity: u8,

    /// List of metrics to use (comma-separated)
    #[arg(long, default_value = "Cs,Ls")]
    metrics: String,

    /// UTF-8 Unicode normalization form
    #[arg(long, default_value = "nfc", value_enum)]
    utf8: NormalizationForm,

    /// Execute calculations sequentially (disable parallel processing)
    #[arg(short, long)]
    sequential: bool,

    /// Pass additional information to evaluation (e.g., 'ignore_geometry')
    #[arg(short = 'x', long)]
    extra: Option<String>,

    /// Language code according to ISO 639-2
    #[arg(short, long, default_value = "deu")]
    language: String,

    /// LanguageTool API URL
    #[arg(short = 'u', long, default_value = "http://localhost:8081")]
    lt_api_url: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate candidates path
    if !args.candidates.is_dir() {
        eprintln!("[ERROR] input \"{}\": invalid directory! exit!", args.candidates.display());
        std::process::exit(1);
    }

    // Validate reference path if provided
    if let Some(ref reference) = args.reference {
        if !reference.is_dir() {
            eprintln!("[ERROR] reference \"{}\": invalid directory! exit!", reference.display());
            std::process::exit(1);
        }

        // Warn if base names don't match
        let base_can = args.candidates.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let base_ref = reference.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if base_can != base_ref {
            eprintln!("[WARN] base domains '{}' and '{}' mismatch, aggregation might fail!", base_can, base_ref);
        }
    }

    // Debug output
    if args.verbosity >= 2 {
        println!("[DEBUG] called with candidates={:?}, reference={:?}, verbosity={}, extra={:?}",
                 args.candidates, args.reference, args.verbosity, args.extra);
    }

    // Initialize metrics
    let metric_list = initialize_metrics(&args.metrics, args.utf8)?;

    if args.verbosity >= 1 {
        println!("[DEBUG] text normalized using '{:?}' code points for '{}'", args.utf8, args.metrics);
    }

    // Create evaluator
    let mut evaluator = Evaluator::new(
        args.candidates.clone(),
        args.verbosity,
        args.extra.clone(),
    );
    evaluator.set_metrics(metric_list);
    evaluator.set_sequential(args.sequential);
    
    if let Some(ref reference) = args.reference {
        evaluator.set_reference(reference.clone());
    }

    // Gather candidates
    let mut candidates = resolve::gather_candidates(&args.candidates)?;
    
    if candidates.is_empty() {
        println!("[WARN] no ocr data (*.xml) in dir starting from '{}'! exit.", args.candidates.display());
        return Ok(());
    }

    // Match groundtruth
    if let Some(ref reference) = args.reference {
        for entry in &mut candidates {
            if let Some(gt) = resolve::find_groundtruth(entry, reference) {
                entry.set_groundtruth(gt);
                entry.align_domains();
            }
        }
    }

    // Filter entries with groundtruth
    let gt_entries: Vec<_> = candidates.iter()
        .filter(|e| e.has_groundtruth())
        .cloned()
        .collect();
    
    let n_entries = candidates.len();
    let n_diff = n_entries - gt_entries.len();

    if args.verbosity >= 1 {
        println!("[DEBUG] from \"{}\" filtered \"{}\" candidates missing groundtruth", n_entries, n_diff);
    }

    // Run evaluation
    evaluator.eval_all(&gt_entries)?;

    // Aggregate results
    evaluator.aggregate(true)?;
    evaluator.eval_map()?;

    // Print report (always print for now)
    evaluator.report_stdout(args.verbosity);

    Ok(())
}

fn initialize_metrics(metrics_str: &str, norm: NormalizationForm) -> Result<Vec<Box<dyn metrics::OCRMetric>>> {
    let tokens: Vec<&str> = metrics_str.split(',').collect();
    let mut metric_objects: Vec<Box<dyn metrics::OCRMetric>> = Vec::new();

    for token in tokens {
        let metric: Box<dyn metrics::OCRMetric> = match token.trim() {
            "Cs" | "Characters" => Box::new(metrics::MetricChars::new(norm)),
            "Ls" | "Letters" => Box::new(metrics::MetricLetters::new(norm)),
            "Ws" | "Words" => Box::new(metrics::MetricWords::new(norm)),
            "BoWs" | "BagOfWords" => Box::new(metrics::MetricBoW::new(norm)),
            "IRPre" | "Pre" | "Precision" => Box::new(metrics::MetricIRPre::new()),
            "IRRec" | "Rec" | "Recall" => Box::new(metrics::MetricIRRec::new()),
            "IRFMeasure" | "FM" => Box::new(metrics::MetricIRFMeasure::new()),
            _ => {
                anyhow::bail!("Unknown metric: '{}'. Available: Cs,Characters,Ls,Letters,Ws,Words,BoWs,BagOfWords,IRPre,Pre,Precision,IRRec,Rec,Recall,IRFMeasure,FM", token);
            }
        };
        metric_objects.push(metric);
    }

    Ok(metric_objects)
}
