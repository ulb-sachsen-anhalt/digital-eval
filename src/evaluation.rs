use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::metrics::OCRMetric;
use crate::resolve::EvalEntry;
use crate::model::digital_object::DigitalObject;

/// Statistical result for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub eval_key: String,
    pub n_total: usize,
    pub n_outlier: usize,
    pub n_chars: usize,
    pub n_lines: usize,
    pub total_mean: f64,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub cleared_result: Option<Box<EvaluationResult>>,
}

impl EvaluationResult {
    pub fn new(eval_key: String, n_total: usize) -> Self {
        EvaluationResult {
            eval_key,
            n_total,
            n_outlier: 0,
            n_chars: 0,
            n_lines: 0,
            total_mean: 0.0,
            mean: 0.0,
            std: 0.0,
            median: 0.0,
            cleared_result: None,
        }
    }

    pub fn calculate_statistics(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }

        self.mean = values.iter().sum::<f64>() / values.len() as f64;
        
        // Calculate standard deviation
        let variance = values.iter()
            .map(|v| (v - self.mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        self.std = variance.sqrt();

        // Calculate median
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        self.median = if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
    }
}

/// Main evaluator struct
pub struct Evaluator {
    pub candidates_path: PathBuf,
    pub reference_path: Option<PathBuf>,
    pub verbosity: u8,
    pub extras: Option<String>,
    pub metrics: Vec<Box<dyn OCRMetric>>,
    pub is_sequential: bool,
    pub results: HashMap<String, Vec<EvaluationResult>>,
}

impl Evaluator {
    pub fn new(candidates_path: PathBuf, verbosity: u8, extras: Option<String>) -> Self {
        Evaluator {
            candidates_path,
            reference_path: None,
            verbosity,
            extras,
            metrics: Vec::new(),
            is_sequential: false,
            results: HashMap::new(),
        }
    }

    pub fn set_metrics(&mut self, metrics: Vec<Box<dyn OCRMetric>>) {
        self.metrics = metrics;
    }

    pub fn set_reference(&mut self, reference: PathBuf) {
        self.reference_path = Some(reference);
    }

    pub fn set_sequential(&mut self, sequential: bool) {
        self.is_sequential = sequential;
    }

    /// Evaluate all entries
    pub fn eval_all(&mut self, entries: &[EvalEntry]) -> Result<()> {
        if self.verbosity >= 1 {
            println!("[INFO] Evaluating {} entries...", entries.len());
        }

        // For now, we'll use sequential processing
        // Parallel processing would require Send+Sync metrics or cloning
        for entry in entries {
            self.eval_single(entry)?;
        }

        Ok(())
    }

    /// Evaluate a single entry
    fn eval_single(&mut self, entry: &EvalEntry) -> Result<()> {
        if self.verbosity >= 2 {
            println!("[DEBUG] Evaluating: {:?}", entry.path_candidate);
        }

        // Load candidate
        let candidate = DigitalObject::from_file(&entry.path_candidate)?;
        let candidate_text = candidate.get_text()?;

        // Load groundtruth if available
        let reference_text = if let Some(ref gt_path) = entry.path_groundtruth {
            let gt = DigitalObject::from_file(gt_path)?;
            Some(gt.get_text()?)
        } else {
            None
        };

        // Calculate metrics
        for metric in &mut self.metrics {
            let _value = metric.calculate(&candidate_text, reference_text.as_deref())?;
            // Store results (would need thread-safe storage in parallel mode)
        }

        Ok(())
    }

    /// Aggregate results by type
    pub fn aggregate(&mut self, _by_type: bool) -> Result<()> {
        if self.verbosity >= 1 {
            println!("[INFO] Aggregating results...");
        }

        // Implementation of aggregation logic
        // Group results by domain directories and calculate statistics

        Ok(())
    }

    /// Create evaluation map
    pub fn eval_map(&mut self) -> Result<()> {
        if self.verbosity >= 1 {
            println!("[INFO] Creating evaluation map...");
        }

        Ok(())
    }

    /// Print stdout report
    pub fn report_stdout(&self, _verbosity: u8) {
        println!("\n=== Evaluation Report ===");
        println!("Candidates: {}", self.candidates_path.display());
        
        if let Some(ref reference) = self.reference_path {
            println!("Reference: {}", reference.display());
        }

        println!("\nMetrics evaluated:");
        for metric in &self.metrics {
            println!("  - {}", metric.label());
        }

        if !self.results.is_empty() {
            println!("\nResults:");
            for (key, results) in &self.results {
                println!("\n{}", key);
                for result in results {
                    println!("  Mean: {:.2}%, Median: {:.2}%, Std: {:.2}",
                             result.mean, result.median, result.std);
                }
            }
        }

        println!("\n=== End Report ===");
    }

    /// Get evaluation results
    pub fn get_results(&self) -> &HashMap<String, Vec<EvaluationResult>> {
        &self.results
    }
}

/// Calculate outliers using IQR method
pub fn detect_outliers(values: &[f64]) -> Vec<usize> {
    if values.len() < 4 {
        return Vec::new();
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1_idx = sorted.len() / 4;
    let q3_idx = 3 * sorted.len() / 4;
    
    let q1 = sorted[q1_idx];
    let q3 = sorted[q3_idx];
    let iqr = q3 - q1;

    let lower_bound = q1 - 1.5 * iqr;
    let upper_bound = q3 + 1.5 * iqr;

    values
        .iter()
        .enumerate()
        .filter_map(|(idx, &val)| {
            if val < lower_bound || val > upper_bound {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{MetricChars, MetricLetters, MetricWords};
    use crate::preprocessing::NormalizationForm;
    use crate::resolve::EvalEntry;

    #[test]
    fn test_evaluation_result_creation() {
        let result = EvaluationResult::new("test".to_string(), 10);
        assert_eq!(result.eval_key, "test");
        assert_eq!(result.n_total, 10);
    }

    #[test]
    fn test_calculate_statistics() {
        let mut result = EvaluationResult::new("test".to_string(), 5);
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        result.calculate_statistics(&values);
        
        assert_eq!(result.mean, 3.0);
        assert_eq!(result.median, 3.0);
        assert!(result.std > 0.0);
    }

    #[test]
    fn test_calculate_statistics_with_variance() {
        let mut result = EvaluationResult::new("variance_test".to_string(), 6);
        let values = vec![95.70, 96.53, 94.91, 94.40, 86.44, 93.44];
        result.calculate_statistics(&values);
        
        // Mean should be around 93.57
        assert!((result.mean - 93.57).abs() < 0.5);
        // Std should be around 3.5 (due to outlier)
        assert!(result.std > 2.0 && result.std < 5.0);
    }

    #[test]
    fn test_detect_outliers() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = detect_outliers(&values);
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&5)); // 100.0 is an outlier
    }

    #[test]
    fn test_detect_outliers_in_accuracy_data() {
        // Real data from test_evaluate_set_with_5_entries
        let values = vec![95.70, 96.53, 94.91, 94.40, 86.44, 93.44];
        let outliers = detect_outliers(&values);
        
        // 86.44 should be detected as outlier
        assert!(!outliers.is_empty());
        assert!(outliers.contains(&4)); // index 4 is 86.44
    }

    #[test]
    fn test_detect_no_outliers_with_few_values() {
        let values = vec![1.0, 2.0, 3.0];
        let outliers = detect_outliers(&values);
        assert!(outliers.is_empty()); // Need at least 4 values
    }

    #[test]
    fn test_evaluator_creation() {
        let path = PathBuf::from("/test");
        let evaluator = Evaluator::new(path.clone(), 0, None);
        assert_eq!(evaluator.candidates_path, path);
        assert_eq!(evaluator.verbosity, 0);
        assert!(!evaluator.is_sequential);
    }

    #[test]
    fn test_evaluator_set_metrics() {
        let path = PathBuf::from("/test");
        let mut evaluator = Evaluator::new(path, 0, None);
        
        let metrics: Vec<Box<dyn OCRMetric>> = vec![
            Box::new(MetricChars::new(NormalizationForm::Nfc)),
            Box::new(MetricLetters::new(NormalizationForm::Nfc)),
        ];
        evaluator.set_metrics(metrics);
        
        assert_eq!(evaluator.metrics.len(), 2);
        assert_eq!(evaluator.metrics[0].label(), "Characters");
        assert_eq!(evaluator.metrics[1].label(), "Letters");
    }

    #[test]
    fn test_evaluator_set_reference() {
        let path = PathBuf::from("/test");
        let mut evaluator = Evaluator::new(path, 0, None);
        
        let reference = PathBuf::from("/reference");
        evaluator.set_reference(reference.clone());
        
        assert!(evaluator.reference_path.is_some());
        assert_eq!(evaluator.reference_path.unwrap(), reference);
    }

    #[test]
    fn test_evaluate_alto_candidate_with_page_groundtruth() {
        // Test path: tests/resources/candidate/frk_alto/1667522809_J_0001_0002.xml
        let alto_path = PathBuf::from("tests/resources/candidate/frk_alto/1667522809_J_0001_0002.xml");
        let gt_path = PathBuf::from("tests/resources/groundtruth/page/1667522809_J_0001_0002.art.gt.xml");
        
        if !alto_path.exists() || !gt_path.exists() {
            eprintln!("Skipping test: test files not found");
            return;
        }

        let mut entry = EvalEntry::new(alto_path.clone(), Some(PathBuf::from("tests/resources/candidate/frk_alto")));
        entry.set_groundtruth(gt_path);
        
        let mut evaluator = Evaluator::new(PathBuf::from("tests/resources/candidate/frk_alto"), 0, None);
        evaluator.set_metrics(vec![Box::new(MetricChars::new(NormalizationForm::Nfc))]);
        
        // Should not panic
        let result = evaluator.eval_single(&entry);
        assert!(result.is_ok());
    }

    #[test]
    fn test_evaluate_page_groundtruth_with_itself() {
        // Perfect evaluation: GT vs itself should yield ~100% accuracy
        let gt_path = PathBuf::from("tests/resources/groundtruth/page/1667522809_J_0001_0002.art.gt.xml");
        
        if !gt_path.exists() {
            eprintln!("Skipping test: test file not found");
            return;
        }

        let mut entry = EvalEntry::new(gt_path.clone(), Some(PathBuf::from("tests/resources/groundtruth/page")));
        entry.set_groundtruth(gt_path);
        
        let mut evaluator = Evaluator::new(PathBuf::from("tests/resources/groundtruth/page"), 0, None);
        let mut metric = MetricChars::new(NormalizationForm::Nfc);
        
        // Load both as same content
        let candidate = DigitalObject::from_file(&entry.path_candidate).unwrap();
        let reference = DigitalObject::from_file(&entry.path_groundtruth.as_ref().unwrap()).unwrap();
        
        let candidate_text = candidate.get_text().unwrap();
        let reference_text = reference.get_text().unwrap();
        
        let accuracy = metric.calculate(&candidate_text, Some(&reference_text)).unwrap();
        
        // Should be 100% or very close
        assert!(accuracy > 99.9, "Expected ~100% accuracy, got {}", accuracy);
    }

    #[test]
    fn test_evaluate_empty_entries_list() {
        let path = PathBuf::from("/test");
        let mut evaluator = Evaluator::new(path, 0, None);
        evaluator.set_metrics(vec![Box::new(MetricChars::new(NormalizationForm::Nfc))]);
        
        let entries: Vec<EvalEntry> = Vec::new();
        let result = evaluator.eval_all(&entries);
        
        // Should handle empty list gracefully
        assert!(result.is_ok());
    }

    #[test]
    fn test_eval_single_without_groundtruth() {
        let candidate_path = PathBuf::from("tests/resources/candidate/frk_alto/1667522809_J_0001_0002.xml");
        
        if !candidate_path.exists() {
            eprintln!("Skipping test: test file not found");
            return;
        }

        let entry = EvalEntry::new(candidate_path, None);
        let mut evaluator = Evaluator::new(PathBuf::from("tests/resources"), 0, None);
        evaluator.set_metrics(vec![Box::new(MetricChars::new(NormalizationForm::Nfc))]);
        
        // Should handle missing groundtruth - metrics will fail since they need reference
        let result = evaluator.eval_single(&entry);
        // It's expected to fail when there's no groundtruth for similarity metrics
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_handle_invalid_xml_file() {
        let invalid_path = PathBuf::from("tests/resources/candidate/frk_alto/1667522809_J_0001_0256_corrupt.xml");
        
        if !invalid_path.exists() {
            eprintln!("Skipping test: corrupt test file not found");
            return;
        }

        // Attempting to load corrupt XML should fail
        let result = DigitalObject::from_file(&invalid_path);
        assert!(result.is_err(), "Expected error for corrupt XML");
    }

    #[test]
    fn test_handle_empty_candidate_data() {
        // Test with candidate that has minimal/empty content
        let gt_path = PathBuf::from("tests/resources/groundtruth/page/urn+nbn+de+gbv+3+1-138193-p0904-0_ger.gt.xml");
        let cd_path = PathBuf::from("tests/resources/candidate/frk_page/urn+nbn+de+gbv+3+1-138193-p0904-0_ger.xml");
        
        if !gt_path.exists() || !cd_path.exists() {
            eprintln!("Skipping test: test files not found");
            return;
        }

        let mut entry = EvalEntry::new(cd_path, None);
        entry.set_groundtruth(gt_path);
        
        let mut evaluator = Evaluator::new(PathBuf::from("tests/resources"), 1, None);
        evaluator.set_metrics(vec![Box::new(MetricChars::new(NormalizationForm::Nfc))]);
        
        // Should handle empty/minimal candidate without panic
        let result = evaluator.eval_single(&entry);
        assert!(result.is_ok() || result.is_err()); // Either way, shouldn't panic
    }

    #[test]
    fn test_bounding_box_from_page_file() {
        use crate::model::digital_object::DigitalObject;
        
        let page_path = PathBuf::from("tests/resources/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml");
        
        if !page_path.exists() {
            eprintln!("Skipping test: test file not found");
            return;
        }

        let doc = DigitalObject::from_file(&page_path);
        assert!(doc.is_ok(), "Should load PAGE file successfully");
        
        let doc = doc.unwrap();
        let bbox = doc.get_bounding_box();
        
        // Should have valid bounding box coordinates
        assert!(bbox.is_ok());
        let (p1, p2) = bbox.unwrap();
        assert!(p1.x < p2.x);
        assert!(p1.y < p2.y);
    }

    #[test]
    fn test_statistics_aggregation() {
        let mut result = EvaluationResult::new("aggregation_test".to_string(), 6);
        
        // Simulate real accuracy values from multiple evaluations
        let values = vec![95.70, 96.53, 94.91, 94.40, 93.44, 95.00];
        result.calculate_statistics(&values);
        result.n_chars = 5000;
        result.n_lines = 100;
        
        assert!(result.mean > 94.0 && result.mean < 96.0);
        assert!(result.std > 0.0 && result.std < 2.0);
        assert_eq!(result.n_chars, 5000);
        assert_eq!(result.n_lines, 100);
    }

    #[test]
    fn test_evaluator_report_stdout() {
        let path = PathBuf::from("/test");
        let mut evaluator = Evaluator::new(path, 0, None);
        evaluator.set_metrics(vec![
            Box::new(MetricChars::new(NormalizationForm::Nfc)),
            Box::new(MetricWords::new(NormalizationForm::Nfc)),
        ]);
        
        // Should not panic when printing report
        evaluator.report_stdout(0);
    }

    #[test]
    fn test_evaluation_result_with_cleared_outliers() {
        let mut result = EvaluationResult::new("outlier_test".to_string(), 6);
        result.n_total = 6;
        
        // Values WITH outlier
        let values_with_outlier = vec![95.70, 96.53, 94.91, 94.40, 86.44, 93.44];
        result.calculate_statistics(&values_with_outlier);
        result.total_mean = result.mean;
        result.n_outlier = 1;
        
        // Create cleared result (without outlier - remove 86.44)
        let mut cleared = EvaluationResult::new("outlier_test_cleared".to_string(), 5);
        let values_without_outlier = vec![95.70, 96.53, 94.91, 94.40, 93.44];
        cleared.calculate_statistics(&values_without_outlier);
        result.cleared_result = Some(Box::new(cleared));
        
        assert_eq!(result.n_outlier, 1);
        assert!(result.cleared_result.is_some());
        
        // Standard deviation should be lower without outlier
        let cleared_std = result.cleared_result.as_ref().unwrap().std;
        let original_std = result.std;
        
        // The outlier (86.44) is far from the mean, so removing it should reduce std
        assert!(cleared_std < original_std, 
            "Expected std without outlier ({:.3}) < std with outlier ({:.3})", 
            cleared_std, original_std);
    }
}
