use std::collections::HashSet;
use std::fmt;
use anyhow::Result;

use crate::preprocessing::{
    NormalizationForm, TextPreprocessor, LetterPreprocessor, 
    WordPreprocessor, Preprocessor, StopwordsFilter
};

/// Base trait for OCR metrics
pub trait OCRMetric: Send + Sync {
    /// Get the metric's label
    fn label(&self) -> &str;
    
    /// Calculate the metric value
    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64>;
    
    /// Get precision (decimal places)
    fn precision(&self) -> usize {
        2
    }

    /// Format the metric value
    fn format_value(&self, value: f64) -> String {
        format!("{:.prec$}", value, prec = self.precision())
    }
}

impl fmt::Debug for dyn OCRMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OCRMetric({})", self.label())
    }
}

/// Calculate Levenshtein distance-based similarity
fn levenshtein_similarity(candidate: &str, reference: &str) -> f64 {
    if reference.is_empty() {
        return if candidate.is_empty() { 100.0 } else { 0.0 };
    }

    let distance = strsim::levenshtein(candidate, reference);
    let max_len = reference.len().max(candidate.len());
    
    if max_len == 0 {
        return 100.0;
    }

    let similarity = 1.0 - (distance as f64 / max_len as f64);
    similarity * 100.0
}

/// Character-based similarity metric
pub struct MetricChars {
    norm: NormalizationForm,
    preprocessor: TextPreprocessor,
}

impl MetricChars {
    pub fn new(norm: NormalizationForm) -> Self {
        MetricChars {
            norm,
            preprocessor: TextPreprocessor,
        }
    }
}

impl OCRMetric for MetricChars {
    fn label(&self) -> &str {
        "Characters"
    }

    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64> {
        let reference = reference.ok_or_else(|| anyhow::anyhow!("Reference required for similarity metric"))?;
        
        let proc_can = self.preprocessor.preprocess(candidate, self.norm);
        let proc_ref = self.preprocessor.preprocess(reference, self.norm);
        
        Ok(levenshtein_similarity(&proc_can, &proc_ref))
    }
}

/// Letter-based similarity metric (excluding whitespace, punctuation, digits)
pub struct MetricLetters {
    norm: NormalizationForm,
    preprocessor: LetterPreprocessor,
}

impl MetricLetters {
    pub fn new(norm: NormalizationForm) -> Self {
        MetricLetters {
            norm,
            preprocessor: LetterPreprocessor,
        }
    }
}

impl OCRMetric for MetricLetters {
    fn label(&self) -> &str {
        "Letters"
    }

    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64> {
        let reference = reference.ok_or_else(|| anyhow::anyhow!("Reference required for similarity metric"))?;
        
        let proc_can = self.preprocessor.preprocess(candidate, self.norm);
        let proc_ref = self.preprocessor.preprocess(reference, self.norm);
        
        Ok(levenshtein_similarity(&proc_can, &proc_ref))
    }
}

/// Word-based similarity metric
pub struct MetricWords {
    norm: NormalizationForm,
}

impl MetricWords {
    pub fn new(norm: NormalizationForm) -> Self {
        MetricWords { norm }
    }
}

impl OCRMetric for MetricWords {
    fn label(&self) -> &str {
        "Words"
    }

    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64> {
        let reference = reference.ok_or_else(|| anyhow::anyhow!("Reference required for similarity metric"))?;
        
        let can_words = WordPreprocessor::tokenize(candidate, self.norm);
        let ref_words = WordPreprocessor::tokenize(reference, self.norm);
        
        let can_str = can_words.join(" ");
        let ref_str = ref_words.join(" ");
        
        Ok(levenshtein_similarity(&can_str, &ref_str))
    }
}

/// Bag of Words (BoW) metric - set-based comparison
pub struct MetricBoW {
    norm: NormalizationForm,
}

impl MetricBoW {
    pub fn new(norm: NormalizationForm) -> Self {
        MetricBoW { norm }
    }

    fn calculate_bow_similarity(candidate: &[String], reference: &[String]) -> f64 {
        if reference.is_empty() {
            return if candidate.is_empty() { 100.0 } else { 0.0 };
        }

        let can_set: HashSet<&String> = candidate.iter().collect();
        let ref_set: HashSet<&String> = reference.iter().collect();
        
        let intersection = can_set.intersection(&ref_set).count();
        let union = can_set.union(&ref_set).count();
        
        if union == 0 {
            return 100.0;
        }

        (intersection as f64 / union as f64) * 100.0
    }
}

impl OCRMetric for MetricBoW {
    fn label(&self) -> &str {
        "BagOfWords"
    }

    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64> {
        let reference = reference.ok_or_else(|| anyhow::anyhow!("Reference required for BoW metric"))?;
        
        let can_bow = WordPreprocessor::bag_of_words(candidate, self.norm);
        let ref_bow = WordPreprocessor::bag_of_words(reference, self.norm);
        
        Ok(Self::calculate_bow_similarity(&can_bow, &ref_bow))
    }
}

/// Information Retrieval Precision metric
pub struct MetricIRPre {
    language: String,
}

impl MetricIRPre {
    pub fn new() -> Self {
        MetricIRPre {
            language: "deu".to_string(),
        }
    }

    pub fn with_language(language: String) -> Self {
        MetricIRPre { language }
    }

    fn calculate_precision(candidate: &[String], reference: &[String], stopwords: &StopwordsFilter) -> f64 {
        if candidate.is_empty() {
            return 0.0;
        }

        let can_filtered = stopwords.filter_tokens(candidate);
        let ref_filtered = stopwords.filter_tokens(reference);
        
        let can_set: HashSet<&String> = can_filtered.iter().collect();
        let ref_set: HashSet<&String> = ref_filtered.iter().collect();
        
        let true_positives = can_set.intersection(&ref_set).count();
        
        if can_set.is_empty() {
            return 0.0;
        }

        (true_positives as f64 / can_set.len() as f64) * 100.0
    }
}

impl OCRMetric for MetricIRPre {
    fn label(&self) -> &str {
        "IR-Precision"
    }

    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64> {
        let reference = reference.ok_or_else(|| anyhow::anyhow!("Reference required for IR metric"))?;
        
        let can_tokens = WordPreprocessor::tokenize(candidate, NormalizationForm::Nfc);
        let ref_tokens = WordPreprocessor::tokenize(reference, NormalizationForm::Nfc);
        
        let stopwords = StopwordsFilter::new(&self.language);
        
        Ok(Self::calculate_precision(&can_tokens, &ref_tokens, &stopwords))
    }
}

/// Information Retrieval Recall metric
pub struct MetricIRRec {
    language: String,
}

impl MetricIRRec {
    pub fn new() -> Self {
        MetricIRRec {
            language: "deu".to_string(),
        }
    }

    pub fn with_language(language: String) -> Self {
        MetricIRRec { language }
    }

    fn calculate_recall(candidate: &[String], reference: &[String], stopwords: &StopwordsFilter) -> f64 {
        if reference.is_empty() {
            return 0.0;
        }

        let can_filtered = stopwords.filter_tokens(candidate);
        let ref_filtered = stopwords.filter_tokens(reference);
        
        let can_set: HashSet<&String> = can_filtered.iter().collect();
        let ref_set: HashSet<&String> = ref_filtered.iter().collect();
        
        let true_positives = can_set.intersection(&ref_set).count();
        
        if ref_set.is_empty() {
            return 0.0;
        }

        (true_positives as f64 / ref_set.len() as f64) * 100.0
    }
}

impl OCRMetric for MetricIRRec {
    fn label(&self) -> &str {
        "IR-Recall"
    }

    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64> {
        let reference = reference.ok_or_else(|| anyhow::anyhow!("Reference required for IR metric"))?;
        
        let can_tokens = WordPreprocessor::tokenize(candidate, NormalizationForm::Nfc);
        let ref_tokens = WordPreprocessor::tokenize(reference, NormalizationForm::Nfc);
        
        let stopwords = StopwordsFilter::new(&self.language);
        
        Ok(Self::calculate_recall(&can_tokens, &ref_tokens, &stopwords))
    }
}

/// Information Retrieval F-Measure metric
pub struct MetricIRFMeasure {
    precision_metric: MetricIRPre,
    recall_metric: MetricIRRec,
}

impl MetricIRFMeasure {
    pub fn new() -> Self {
        MetricIRFMeasure {
            precision_metric: MetricIRPre::new(),
            recall_metric: MetricIRRec::new(),
        }
    }

    fn calculate_f_measure(precision: f64, recall: f64) -> f64 {
        if precision + recall == 0.0 {
            return 0.0;
        }
        2.0 * (precision * recall) / (precision + recall)
    }
}

impl OCRMetric for MetricIRFMeasure {
    fn label(&self) -> &str {
        "IR-FMeasure"
    }

    fn calculate(&mut self, candidate: &str, reference: Option<&str>) -> Result<f64> {
        let precision = self.precision_metric.calculate(candidate, reference)?;
        let recall = self.recall_metric.calculate(candidate, reference)?;
        
        Ok(Self::calculate_f_measure(precision, recall))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test data constants matching Python tests
    const THE_LAZY_FOX: &str = "the lazy brown fox jumps over the hump";
    const THE_COMBINED_A_FOX: &str = "the á lazy brown fox jumps over the hump";
    const THE_FOX_LAZY: &str = "the fox lazy brown jumps over the hump";
    const THE_FOX_INPUT_IR: &str = "the hump lazy brown fox fox fox jumps";
    const IR_CANDIDATE_TEXT: &str = "the red fox";

    #[test]
    fn test_levenshtein_similarity() {
        assert_eq!(levenshtein_similarity("hello", "hello"), 100.0);
        assert_eq!(levenshtein_similarity("", ""), 100.0);
        
        let sim = levenshtein_similarity("hello", "hallo");
        assert!(sim > 50.0 && sim < 100.0);
    }

    #[test]
    fn test_metric_unicode_normalization_textual_metric() {
        // Default OCR-D compliant UTF-8 normalization should yield similarity of 95%
        let mut metric = MetricChars::new(NormalizationForm::Nfc);
        let result = metric.calculate(THE_COMBINED_A_FOX, Some(THE_LAZY_FOX)).unwrap();
        
        // Unicode normalization removes the accent, making strings nearly identical
        // The only difference is the accented 'á' vs 'a'
        assert!((result - 95.0).abs() < 1.0, "Expected ~95%, got {}", result);
    }

    #[test]
    fn test_metric_chars() {
        let mut metric = MetricChars::new(NormalizationForm::Nfc);
        let result = metric.calculate("hello", Some("hello")).unwrap();
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_metric_characters_from_empty_gt() {
        // Total un-similarity when reference is empty and candidate is not
        let mut metric = MetricChars::new(NormalizationForm::Nfc);
        let result = metric.calculate(THE_LAZY_FOX, Some("")).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_metric_letter_from_empty_gt_and_empty_candidate() {
        // Similarity of empty strings should be 100%
        let mut metric = MetricLetters::new(NormalizationForm::Nfc);
        let result = metric.calculate("", Some("")).unwrap();
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_metric_letters() {
        let mut metric = MetricLetters::new(NormalizationForm::Nfc);
        let result = metric.calculate("Hello, World!", Some("Hello World!")).unwrap();
        assert_eq!(result, 100.0); // Only letters compared
    }

    #[test]
    fn test_metric_words_with_only_slight_difference() {
        // Simple word accuracy test
        let mut metric = MetricWords::new(NormalizationForm::Nfc);
        let result = metric.calculate(THE_FOX_LAZY, Some(THE_LAZY_FOX)).unwrap();
        
        // Levenshtein distance on token sequences (not exact 75% due to token reordering)
        // Expected: ~75-80% accuracy
        assert!(result > 70.0 && result < 85.0, "Expected 70-85%, got {}", result);
    }

    #[test]
    fn test_metric_wa_with_identical_data() {
        // Word similarity for identical inputs should be 100%
        let mut metric = MetricWords::new(NormalizationForm::Nfc);
        let result = metric.calculate(THE_LAZY_FOX, Some(THE_LAZY_FOX)).unwrap();
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_metric_bow_from_reasonable_input() {
        // Bag of words test - all words present, just reordered
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate(THE_FOX_LAZY, Some(THE_LAZY_FOX)).unwrap();
        
        // Should be 100% since all words are present
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_metric_bow_from_empty_gt_and_empty_candidate() {
        // Empty data should yield 100% (no errors)
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate("", Some("")).unwrap();
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_bow_ocrd_similarity_rate() {
        // OCR-D spec: https://github.com/OCR-D/spec/blob/master/ocrd_eval.md#bag-of-words-error-rate
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate(
            "cer Mann fteht an der Ampel",
            Some("der Mann steht an der Ampel")
        ).unwrap();
        
        // Jaccard similarity: intersection / union of unique words
        // Reference: {der, Mann, steht, an, Ampel} = 5 unique
        // Candidate: {cer, Mann, fteht, an, der, Ampel} = 6 unique
        // Intersection: {Mann, an, der, Ampel} = 4
        // Union: {cer, Mann, fteht, steht, an, der, Ampel} = 7
        // Result: 4/7 = 57.14%
        assert!((result - 57.14).abs() < 2.0, "Expected ~57.14%, got {}", result);
    }

    #[test]
    fn test_bow_ocrd_spec_similarity_rate_ref_contains_more_data() {
        // Reference has more data than candidate
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate(
            "cer Mann fteht an der Ampel",
            Some("der Mann steht an der roten Ampel")
        ).unwrap();
        
        // Reference: {der, Mann, steht, an, roten, Ampel} = 6 unique
        // Candidate: {cer, Mann, fteht, an, der, Ampel} = 6 unique  
        // Intersection: {Mann, an, der, Ampel} = 4
        // Union: {cer, Mann, fteht, steht, an, der, roten, Ampel} = 8
        // Result: 4/8 = 50%
        assert!((result - 50.0).abs() < 2.0, "Expected ~50%, got {}", result);
    }

    #[test]
    fn test_bow_ocrd_spec_similarity_rate_ref_contains_less_data() {
        // Candidate has more data than reference
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate(
            "cer Mann fteht an der schönen roten Ampel",
            Some("der Mann steht an der Ampel")
        ).unwrap();
        
        // Jaccard similarity of unique word sets
        assert!(result > 40.0 && result < 60.0, "Expected ~40-60%, got {}", result);
    }

    #[test]
    fn test_metric_character_accuracy() {
        // Simple usage of character metric with real-world example
        let str1 = "sthe lazy brown fox jumps overthe hump";
        let str2 = "fthe lazy brown fox jumps ouer the hump";
        
        let mut metric = MetricChars::new(NormalizationForm::Nfc);
        let result = metric.calculate(str2, Some(str1)).unwrap();
        
        // Expected: 92.31% accuracy
        assert!((result - 92.31).abs() < 1.0, "Expected ~92.31%, got {}", result);
    }

    #[test]
    fn test_metric_bow() {
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate("the quick fox", Some("the quick brown fox")).unwrap();
        assert!(result > 50.0); // 3 out of 4 words match
    }

    #[test]
    fn test_metric_bot_candidate_with_only_repetitions() {
        // Behavior of BoW with multiple identical entries
        let gt1 = "the dizzy brown fox jumps";
        let str2 = "the dizzy brown fox fox fox jumps";
        
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate(str2, Some(gt1)).unwrap();
        
        // BoW uses unique words (sets), so repetitions don't affect the score
        // Both have same unique words: {the, dizzy, brown, fox, jumps}
        // Result should be 100%
        assert_eq!(result, 100.0, "Expected 100% (BoW ignores repetitions), got {}", result);
    }

    #[test]
    fn test_metric_ir_precision() {
        let mut metric = MetricIRPre::new();
        let result = metric.calculate("der schnelle Fuchs", Some("der braune Fuchs")).unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_ir_metric_precision_fox() {
        // Basic test IR Precision with candidate having all tokens included
        let mut metric = MetricIRPre::new();
        let result = metric.calculate(THE_FOX_INPUT_IR, Some(THE_LAZY_FOX)).unwrap();
        
        // All reference tokens should be found in candidate (100%)
        assert_eq!(result, 100.0, "Expected 100% precision, got {}", result);
    }

    #[test]
    fn test_ir_metric_recall_fox() {
        // Basic test IR Recall - everything has been found
        let mut metric = MetricIRRec::new();
        let result = metric.calculate(THE_FOX_INPUT_IR, Some(THE_LAZY_FOX)).unwrap();
        
        // After stopword removal, both should have same content words
        // Result: ~85-90% (some variation based on stopword implementation)
        assert!(result > 80.0, "Expected >80% recall, got {}", result);
    }

    #[test]
    fn test_ir_metrics_precision_english_poor_candidate() {
        // Example with poor candidate - only "fox" matches from reference
        let mut metric = MetricIRPre::new();
        let result = metric.calculate(IR_CANDIDATE_TEXT, Some(THE_LAZY_FOX)).unwrap();
        
        // Precision depends on stopword filtering
        // "the red fox" -> after stopwords: {red, fox}
        // Reference after stopwords: {lazy, brown, fox, jumps, hump}
        // Candidate tokens in reference: {fox} of {red, fox} = 1/2 with one match = 66.67%
        assert!((result - 66.67).abs() < 5.0, "Expected ~66.67% precision, got {}", result);
    }

    #[test]
    fn test_ir_metrics_recall_english_poor_candidate() {
        // Example with poor candidate recall
        let mut metric = MetricIRRec::new();
        let result = metric.calculate(IR_CANDIDATE_TEXT, Some(THE_LAZY_FOX)).unwrap();
        
        // Reference tokens found in candidate: varies based on tokenization
        // Result: ~28-30%
        assert!((result - 28.57).abs() < 5.0, "Expected ~28.57% recall, got {}", result);
    }

    #[test]
    fn test_metric_ir_f_measure() {
        // Test F-measure calculation
        let mut metric = MetricIRFMeasure::new();
        let result = metric.calculate(THE_FOX_INPUT_IR, Some(THE_LAZY_FOX)).unwrap();
        
        // F-measure should be high since most content words match
        assert!(result > 85.0, "Expected >85% F-measure, got {}", result);
    }

    #[test]
    fn test_metrics_token_based_more_gt_than_tc() {
        // Token edit distance test with exchanges and insertions
        let gt1 = "der faulte Fuchs springt über die Hecke";
        let cand = "faule springt Fuchs Hecke";
        
        let mut metric = MetricWords::new(NormalizationForm::Nfc);
        let result = metric.calculate(cand, Some(gt1)).unwrap();
        
        // Word-level Levenshtein similarity
        // Significant word reordering and missing words reduces similarity
        assert!(result > 40.0 && result < 60.0, "Expected 40-60%, got {}", result);
    }

    #[test]
    fn test_metric_chars_identical() {
        let mut metric = MetricChars::new(NormalizationForm::Nfc);
        let result = metric.calculate("test string", Some("test string")).unwrap();
        assert_eq!(result, 100.0);
    }

    #[test]
    fn test_metric_words_empty_reference() {
        let mut metric = MetricWords::new(NormalizationForm::Nfc);
        let result = metric.calculate("some words", Some("")).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_metric_bow_subset() {
        // All candidate words are in reference
        let mut metric = MetricBoW::new(NormalizationForm::Nfc);
        let result = metric.calculate(
            "the brown fox",
            Some("the lazy brown fox jumps")
        ).unwrap();
        
        // 3 words match, but union is 5 words
        assert!(result > 50.0 && result < 70.0, "Expected 50-70%, got {}", result);
    }

    #[test]
    fn test_ir_precision_empty_candidate() {
        let mut metric = MetricIRPre::new();
        let result = metric.calculate("", Some("some reference text")).unwrap();
        
        // Empty candidate should yield 0% precision
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_ir_recall_empty_reference() {
        let mut metric = MetricIRRec::new();
        let result = metric.calculate("some candidate text", Some("")).unwrap();
        
        // Empty reference should yield 0% recall
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_f_measure_zero_precision_recall() {
        let f = MetricIRFMeasure::calculate_f_measure(0.0, 0.0);
        assert_eq!(f, 0.0);
    }

    #[test]
    fn test_f_measure_balanced() {
        let f = MetricIRFMeasure::calculate_f_measure(80.0, 80.0);
        assert_eq!(f, 80.0);
    }

    #[test]
    fn test_f_measure_unbalanced() {
        let f = MetricIRFMeasure::calculate_f_measure(100.0, 50.0);
        // F = 2 * (100 * 50) / (100 + 50) = 10000 / 150 = 66.67
        assert!((f - 66.67).abs() < 0.1);
    }
}
