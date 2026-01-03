use anyhow::Result;
use std::fs;
use std::path::Path;
use roxmltree::Document;

use super::alto_parser;
use super::page_parser;
use crate::geometry::BoundingBox;

#[allow(dead_code)]

/// Enum for different OCR format types
#[derive(Debug, Clone, PartialEq)]
pub enum FormatType {
    Alto,
    Page,
    Text,
    Unknown,
}

/// Represents a digital object (document) with OCR data
#[derive(Debug)]
pub struct DigitalObject {
    pub format_type: FormatType,
    pub file_path: Option<String>,
    pub text_content: String,
    pub regions: Vec<Region>,
}

/// Represents a text region in a document
#[derive(Debug, Clone)]
pub struct Region {
    pub id: Option<String>,
    pub text: String,
    pub bounding_box: Option<BoundingBox>,
    pub lines: Vec<TextLine>,
}

/// Represents a text line within a region
#[derive(Debug, Clone)]
pub struct TextLine {
    pub id: Option<String>,
    pub text: String,
    pub bounding_box: Option<BoundingBox>,
    pub words: Vec<Word>,
}

/// Represents a word within a text line
#[derive(Debug, Clone)]
pub struct Word {
    pub text: String,
    pub confidence: Option<f64>,
    pub bounding_box: Option<BoundingBox>,
}

impl DigitalObject {
    /// Create a new empty digital object
    pub fn new(format_type: FormatType) -> Self {
        DigitalObject {
            format_type,
            file_path: None,
            text_content: String::new(),
            regions: Vec::new(),
        }
    }

    /// Load a digital object from a file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let file_path = path.to_str().map(|s| s.to_string());

        // Determine format type
        let format_type = Self::detect_format(&content);

        match format_type {
            FormatType::Alto => Self::from_alto(&content, file_path),
            FormatType::Page => Self::from_page(&content, file_path),
            FormatType::Text => Self::from_text(&content, file_path),
            FormatType::Unknown => {
                anyhow::bail!("Unknown or unsupported format: {}", path.display());
            }
        }
    }

    /// Detect the format of the OCR data
    fn detect_format(content: &str) -> FormatType {
        if content.contains("<alto") || content.contains("alto-") {
            FormatType::Alto
        } else if content.contains("<PcGts") || content.contains("page-") {
            FormatType::Page
        } else if content.starts_with("<?xml") {
            // Generic XML, might be ALTO or PAGE
            if content.contains("String") && content.contains("CONTENT") {
                FormatType::Alto
            } else if content.contains("TextLine") && content.contains("TextEquiv") {
                FormatType::Page
            } else {
                FormatType::Unknown
            }
        } else {
            // Assume plain text
            FormatType::Text
        }
    }

    /// Parse ALTO format
    fn from_alto(content: &str, file_path: Option<String>) -> Result<Self> {
        let doc = Document::parse(content)?;
        let (text, regions) = alto_parser::parse_alto_document(&doc)?;

        Ok(DigitalObject {
            format_type: FormatType::Alto,
            file_path,
            text_content: text,
            regions,
        })
    }

    /// Parse PAGE format
    fn from_page(content: &str, file_path: Option<String>) -> Result<Self> {
        let doc = Document::parse(content)?;
        let (text, regions) = page_parser::parse_page_document(&doc)?;

        Ok(DigitalObject {
            format_type: FormatType::Page,
            file_path,
            text_content: text,
            regions,
        })
    }

    /// Parse plain text format
    fn from_text(content: &str, file_path: Option<String>) -> Result<Self> {
        Ok(DigitalObject {
            format_type: FormatType::Text,
            file_path,
            text_content: content.to_string(),
            regions: Vec::new(),
        })
    }

    /// Get the full text content
    pub fn get_text(&self) -> Result<String> {
        Ok(self.text_content.clone())
    }

    /// Get text from a specific region
    pub fn get_region_text(&self, region_id: &str) -> Option<String> {
        self.regions
            .iter()
            .find(|r| r.id.as_deref() == Some(region_id))
            .map(|r| r.text.clone())
    }

    /// Filter regions by bounding box intersection
    pub fn filter_by_area(&self, area: &BoundingBox) -> Vec<&Region> {
        self.regions
            .iter()
            .filter(|region| {
                if let Some(ref bbox) = region.bounding_box {
                    crate::geometry::intersection_area(bbox, area) > 0.0
                } else {
                    false
                }
            })
            .collect()
    }

    /// Get the overall bounding box of the document (union of all regions)
    pub fn get_bounding_box(&self) -> Result<(crate::geometry::Coordinate, crate::geometry::Coordinate)> {
        if self.regions.is_empty() {
            anyhow::bail!("No regions found in document");
        }

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for region in &self.regions {
            if let Some(ref bbox) = region.bounding_box {
                min_x = min_x.min(bbox.min_x);
                min_y = min_y.min(bbox.min_y);
                max_x = max_x.max(bbox.max_x);
                max_y = max_y.max(bbox.max_y);
            }
        }

        if min_x == f64::MAX || min_y == f64::MAX {
            anyhow::bail!("No valid bounding boxes found in regions");
        }

        Ok((
            crate::geometry::Coordinate::new(min_x, min_y), 
            crate::geometry::Coordinate::new(max_x, max_y)
        ))
    }

    /// Get statistics about the document
    pub fn get_statistics(&self) -> DocumentStatistics {
        let n_regions = self.regions.len();
        let n_lines: usize = self.regions.iter().map(|r| r.lines.len()).sum();
        let n_words: usize = self.regions
            .iter()
            .flat_map(|r| &r.lines)
            .map(|l| l.words.len())
            .sum();
        let n_chars = self.text_content.len();

        DocumentStatistics {
            n_regions,
            n_lines,
            n_words,
            n_chars,
        }
    }
}

/// Document statistics
#[derive(Debug, Clone)]
pub struct DocumentStatistics {
    pub n_regions: usize,
    pub n_lines: usize,
    pub n_words: usize,
    pub n_chars: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_detect_format_text() {
        let content = "This is plain text";
        assert_eq!(DigitalObject::detect_format(content), FormatType::Text);
    }

    #[test]
    fn test_detect_format_alto() {
        let content = r#"<?xml version="1.0"?><alto xmlns="http://www.loc.gov/standards/alto/ns-v3#"></alto>"#;
        assert_eq!(DigitalObject::detect_format(content), FormatType::Alto);
    }

    #[test]
    fn test_from_text() {
        let content = "Hello World";
        let obj = DigitalObject::from_text(content, None).unwrap();
        assert_eq!(obj.format_type, FormatType::Text);
        assert_eq!(obj.text_content, "Hello World");
    }

    #[test]
    fn test_new_digital_object() {
        let obj = DigitalObject::new(FormatType::Text);
        assert_eq!(obj.format_type, FormatType::Text);
        assert!(obj.text_content.is_empty());
        assert!(obj.regions.is_empty());
    }

    #[test]
    fn test_get_text() {
        let mut obj = DigitalObject::new(FormatType::Text);
        obj.text_content = "Test content".to_string();
        let text = obj.get_text().unwrap();
        assert_eq!(text, "Test content");
    }

    #[test]
    fn test_document_statistics() {
        let mut obj = DigitalObject::new(FormatType::Text);
        obj.text_content = "Hello World".to_string();
        
        // Add a region with lines and words
        let region = Region {
            id: Some("r1".to_string()),
            text: "Hello World".to_string(),
            bounding_box: None,
            lines: vec![TextLine {
                id: Some("l1".to_string()),
                text: "Hello World".to_string(),
                bounding_box: None,
                words: vec![
                    Word {
                        text: "Hello".to_string(),
                        confidence: None,
                        bounding_box: None,
                    },
                    Word {
                        text: "World".to_string(),
                        confidence: None,
                        bounding_box: None,
                    },
                ],
            }],
        };
        obj.regions.push(region);

        let stats = obj.get_statistics();
        assert_eq!(stats.n_regions, 1);
        assert_eq!(stats.n_lines, 1);
        assert_eq!(stats.n_words, 2);
        assert_eq!(stats.n_chars, 11);
    }

    #[test]
    fn test_alto_groundtruth_file() {
        // Test with actual ALTO groundtruth file if it exists
        let test_file = PathBuf::from("tests/resources/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml");
        
        if test_file.exists() {
            let obj = DigitalObject::from_file(&test_file).unwrap();
            assert_eq!(obj.format_type, FormatType::Alto);
            assert!(!obj.text_content.is_empty());
            assert!(!obj.regions.is_empty());
            
            // Should have 10 regions as per Python test
            assert_eq!(obj.regions.len(), 10);
            
            // Check first region ID
            if let Some(ref id) = obj.regions[0].id {
                assert_eq!(id, "block_27");
            }
            
            // Check second region has 2 lines
            assert_eq!(obj.regions[1].lines.len(), 2);
            
            // Check first line of first region has text
            let first_line_text = &obj.regions[0].lines[0].text;
            assert!(first_line_text.contains("Neueſte") || first_line_text.contains("Neueste"));
        }
    }

    #[test]
    fn test_page_groundtruth_file() {
        // Test with actual PAGE groundtruth file
        let test_file = PathBuf::from("tests/resources/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml");
        
        if test_file.exists() {
            let obj = DigitalObject::from_file(&test_file).unwrap();
            assert_eq!(obj.format_type, FormatType::Page);
            assert!(!obj.text_content.is_empty());
            assert!(!obj.regions.is_empty());
            
            // Should have 1 region as per Python test
            assert_eq!(obj.regions.len(), 1);
            
            // Should have 23 lines
            assert_eq!(obj.regions[0].lines.len(), 23);
            
            // First line should have specific text
            let first_line = &obj.regions[0].lines[0].text;
            assert!(first_line.contains("Schrift"));
        }
    }

    #[test]
    fn test_region_with_bounding_box() {
        let bbox = BoundingBox::new(100.0, 200.0, 600.0, 300.0);
        let region = Region {
            id: Some("test_region".to_string()),
            text: "Test text".to_string(),
            bounding_box: Some(bbox),
            lines: vec![],
        };

        assert!(region.bounding_box.is_some());
        let bb = region.bounding_box.unwrap();
        assert_eq!(bb.width(), 500.0);
        assert_eq!(bb.height(), 100.0);
    }

    #[test]
    fn test_word_with_confidence() {
        let word = Word {
            text: "test".to_string(),
            confidence: Some(0.95),
            bounding_box: None,
        };

        assert_eq!(word.text, "test");
        assert_eq!(word.confidence, Some(0.95));
    }

    #[test]
    fn test_filter_by_area() {
        let mut obj = DigitalObject::new(FormatType::Alto);
        
        // Add regions with bounding boxes
        let region1 = Region {
            id: Some("r1".to_string()),
            text: "Region 1".to_string(),
            bounding_box: Some(BoundingBox::new(0.0, 0.0, 100.0, 100.0)),
            lines: vec![],
        };
        
        let region2 = Region {
            id: Some("r2".to_string()),
            text: "Region 2".to_string(),
            bounding_box: Some(BoundingBox::new(200.0, 200.0, 300.0, 300.0)),
            lines: vec![],
        };
        
        obj.regions.push(region1);
        obj.regions.push(region2);

        // Filter by area that overlaps with region1 only
        let filter_area = BoundingBox::new(50.0, 50.0, 150.0, 150.0);
        let filtered = obj.filter_by_area(&filter_area);
        
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id.as_deref(), Some("r1"));
    }

    #[test]
    fn test_get_region_text() {
        let mut obj = DigitalObject::new(FormatType::Alto);
        
        let region = Region {
            id: Some("test_region".to_string()),
            text: "Test region text".to_string(),
            bounding_box: None,
            lines: vec![],
        };
        
        obj.regions.push(region);

        let text = obj.get_region_text("test_region");
        assert_eq!(text, Some("Test region text".to_string()));
        
        let missing = obj.get_region_text("nonexistent");
        assert_eq!(missing, None);
    }

    #[test]
    fn test_format_detection_page() {
        let content = r#"<?xml version="1.0"?><PcGts xmlns="http://schema.primaresearch.org/PAGE"></PcGts>"#;
        assert_eq!(DigitalObject::detect_format(content), FormatType::Page);
    }

    #[test]
    fn test_empty_regions() {
        let obj = DigitalObject::new(FormatType::Alto);
        let stats = obj.get_statistics();
        
        assert_eq!(stats.n_regions, 0);
        assert_eq!(stats.n_lines, 0);
        assert_eq!(stats.n_words, 0);
        assert_eq!(stats.n_chars, 0);
    }
}
