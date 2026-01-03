use anyhow::Result;
use roxmltree::{Document, Node};
use std::collections::HashMap;

use super::digital_object::{Region, TextLine, Word};
use crate::geometry::{BoundingBox, parse_polygon_string};

/// Parse a PAGE XML document
pub fn parse_page_document(doc: &Document) -> Result<(String, Vec<Region>)> {
    let mut full_text = String::new();
    let mut regions = Vec::new();

    // First, check if there's a reading order
    let reading_order = extract_reading_order(doc);

    // Collect all regions with their IDs
    let mut region_map: HashMap<String, Region> = HashMap::new();
    let mut region_dom_order = Vec::new();

    for node in doc.descendants() {
        if node.has_tag_name("TextRegion") {
            if let Ok(region) = parse_text_region(&node) {
                if let Some(ref id) = region.id {
                    region_dom_order.push(id.clone());
                    region_map.insert(id.clone(), region);
                } else {
                    // Region without ID, add directly
                    regions.push(region);
                }
            }
        }
    }

    // Apply reading order if available, otherwise use DOM order
    let ordered_ids = if !reading_order.is_empty() {
        apply_reading_order(&reading_order, &region_dom_order)
    } else {
        region_dom_order
    };

    // Build final regions list in correct order
    for id in ordered_ids {
        if let Some(region) = region_map.remove(&id) {
            full_text.push_str(&region.text);
            full_text.push('\n');
            regions.push(region);
        }
    }

    Ok((full_text.trim().to_string(), regions))
}

/// Extract reading order from PAGE document
fn extract_reading_order(doc: &Document) -> HashMap<String, usize> {
    let mut reading_order = HashMap::new();

    for node in doc.descendants() {
        if node.has_tag_name("RegionRefIndexed") {
            if let (Some(region_ref), Some(index_str)) = 
                (node.attribute("regionRef"), node.attribute("index")) {
                if let Ok(index) = index_str.parse::<usize>() {
                    reading_order.insert(region_ref.to_string(), index);
                }
            }
        }
    }

    reading_order
}

/// Apply reading order to region IDs
fn apply_reading_order(
    reading_order: &HashMap<String, usize>,
    dom_order: &[String]
) -> Vec<String> {
    let mut ordered_regions: Vec<(String, usize)> = Vec::new();
    let mut unordered_regions = Vec::new();

    for id in dom_order {
        if let Some(&index) = reading_order.get(id) {
            ordered_regions.push((id.clone(), index));
        } else {
            unordered_regions.push(id.clone());
        }
    }

    // Sort by index
    ordered_regions.sort_by_key(|(_, index)| *index);

    // Combine: ordered regions first, then unordered ones
    let mut result: Vec<String> = ordered_regions.into_iter()
        .map(|(id, _)| id)
        .collect();
    result.extend(unordered_regions);

    result
}

/// Parse a TextRegion element
fn parse_text_region(node: &Node) -> Result<Region> {
    let id = node.attribute("id").map(|s| s.to_string());
    let bounding_box = parse_coords(node);

    let mut lines = Vec::new();
    let mut region_text = String::new();

    // Find all TextLine elements
    for child in node.descendants() {
        if child.has_tag_name("TextLine") {
            if let Ok(line) = parse_text_line(&child) {
                region_text.push_str(&line.text);
                region_text.push('\n');
                lines.push(line);
            }
        }
    }

    Ok(Region {
        id,
        text: region_text.trim().to_string(),
        bounding_box,
        lines,
    })
}

/// Parse a TextLine element
fn parse_text_line(node: &Node) -> Result<TextLine> {
    let id = node.attribute("id").map(|s| s.to_string());
    let bounding_box = parse_coords(node);

    let mut words = Vec::new();
    let mut line_text = String::new();

    // Find TextEquiv/Unicode for line text
    for child in node.descendants() {
        if child.has_tag_name("Unicode") {
            if let Some(text) = child.text() {
                line_text = text.to_string();
            }
        }
    }

    // Find all Word elements
    for child in node.descendants() {
        if child.has_tag_name("Word") {
            if let Ok(word) = parse_word(&child) {
                words.push(word);
            }
        }
    }

    Ok(TextLine {
        id,
        text: line_text,
        bounding_box,
        words,
    })
}

/// Parse a Word element
fn parse_word(node: &Node) -> Result<Word> {
    let bounding_box = parse_coords(node);
    let mut text = String::new();
    let mut confidence = None;

    // Find TextEquiv/Unicode for word text
    for child in node.descendants() {
        if child.has_tag_name("Unicode") {
            if let Some(t) = child.text() {
                text = t.to_string();
            }
        }
    }

    // Try to get confidence
    if let Some(conf_str) = node.attribute("conf") {
        confidence = conf_str.parse::<f64>().ok();
    }

    Ok(Word {
        text,
        confidence,
        bounding_box,
    })
}

/// Parse Coords element to get bounding box
fn parse_coords(node: &Node) -> Option<BoundingBox> {
    for child in node.children() {
        if child.has_tag_name("Coords") {
            if let Some(points) = child.attribute("points") {
                if let Ok(coords) = parse_polygon_string(points) {
                    if let Ok(bbox) = BoundingBox::from_points(&coords) {
                        return Some(bbox);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_simple_page() {
        let xml = r#"<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page>
        <TextRegion id="r1">
            <Coords points="100,100 500,100 500,200 100,200"/>
            <TextLine id="l1">
                <Coords points="100,100 500,100 500,150 100,150"/>
                <TextEquiv>
                    <Unicode>Hello World</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_page_document(&doc).unwrap();
        
        assert!(text.contains("Hello World"));
        assert!(!regions.is_empty());
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].id.as_deref(), Some("r1"));
    }

    #[test]
    fn test_parse_coords() {
        let xml = r#"<?xml version="1.0"?>
<Element>
    <Coords points="100,100 200,100 200,200 100,200"/>
</Element>"#;

        let doc = Document::parse(xml).unwrap();
        let node = doc.root_element();
        let bbox = parse_coords(&node);
        
        assert!(bbox.is_some());
        let bbox = bbox.unwrap();
        assert_eq!(bbox.min_x, 100.0);
        assert_eq!(bbox.max_x, 200.0);
        assert_eq!(bbox.min_y, 100.0);
        assert_eq!(bbox.max_y, 200.0);
    }

    #[test]
    fn test_parse_page_multiple_regions() {
        let xml = r#"<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page>
        <TextRegion id="region1">
            <Coords points="0,0 100,0 100,50 0,50"/>
            <TextLine id="line1">
                <Coords points="0,0 100,0 100,25 0,25"/>
                <TextEquiv><Unicode>First Region</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
        <TextRegion id="region2">
            <Coords points="0,60 100,60 100,110 0,110"/>
            <TextLine id="line2">
                <Coords points="0,60 100,60 100,85 0,85"/>
                <TextEquiv><Unicode>Second Region</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_page_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 2);
        assert!(text.contains("First Region"));
        assert!(text.contains("Second Region"));
        assert_eq!(regions[0].id.as_deref(), Some("region1"));
        assert_eq!(regions[1].id.as_deref(), Some("region2"));
    }

    #[test]
    fn test_parse_page_with_words() {
        let xml = r#"<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page>
        <TextRegion id="r1">
            <Coords points="0,0 200,0 200,100 0,100"/>
            <TextLine id="l1">
                <Coords points="0,0 200,0 200,50 0,50"/>
                <Word id="w1" conf="0.95">
                    <Coords points="0,0 80,0 80,50 0,50"/>
                    <TextEquiv><Unicode>Hello</Unicode></TextEquiv>
                </Word>
                <Word id="w2" conf="0.98">
                    <Coords points="90,0 180,0 180,50 90,50"/>
                    <TextEquiv><Unicode>World</Unicode></TextEquiv>
                </Word>
                <TextEquiv><Unicode>Hello World</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_page_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].lines.len(), 1);
        assert_eq!(regions[0].lines[0].words.len(), 2);
        
        assert_eq!(regions[0].lines[0].words[0].text, "Hello");
        assert_eq!(regions[0].lines[0].words[1].text, "World");
        assert_eq!(regions[0].lines[0].words[0].confidence, Some(0.95));
    }

    #[test]
    fn test_page_groundtruth_odem() {
        let test_file = PathBuf::from("tests/resources/groundtruth/page/urn+nbn+de+gbv+3+1-115907-p0042-0_ger.gt.xml");
        
        if test_file.exists() {
            let content = std::fs::read_to_string(&test_file).unwrap();
            let doc = Document::parse(&content).unwrap();
            let (text, regions) = parse_page_document(&doc).unwrap();
            
            // Based on Python test: 1 region
            assert_eq!(regions.len(), 1);
            
            // 23 lines
            assert_eq!(regions[0].lines.len(), 23);
            
            // Check first line text
            let first_line = &regions[0].lines[0].text;
            assert!(first_line.contains("Schrift") || first_line.contains("erklaͤret"));
            
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn test_page_multiple_lines() {
        let xml = r#"<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page>
        <TextRegion id="r1">
            <Coords points="0,0 300,0 300,150 0,150"/>
            <TextLine id="l1">
                <Coords points="0,0 300,0 300,50 0,50"/>
                <TextEquiv><Unicode>First line</Unicode></TextEquiv>
            </TextLine>
            <TextLine id="l2">
                <Coords points="0,60 300,60 300,110 0,110"/>
                <TextEquiv><Unicode>Second line</Unicode></TextEquiv>
            </TextLine>
            <TextLine id="l3">
                <Coords points="0,120 300,120 300,150 0,150"/>
                <TextEquiv><Unicode>Third line</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_page_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].lines.len(), 3);
        assert!(text.contains("First line"));
        assert!(text.contains("Second line"));
        assert!(text.contains("Third line"));
    }

    #[test]
    fn test_reading_order_respected() {
        // Test that reading order from RegionRefIndexed is respected
        // The reading order specifies: index 0: region_003, index 1: region_001, index 2: region_002
        // while the DOM order is: region_001, region_002, region_003
        let test_file = PathBuf::from("tests/resources/groundtruth/page/reading_order_respected.xml");
        
        if !test_file.exists() {
            eprintln!("Skipping test: test file not found");
            return;
        }

        let content = std::fs::read_to_string(&test_file).unwrap();
        let doc = Document::parse(&content).unwrap();
        let (_text, regions) = parse_page_document(&doc).unwrap();

        // Assert - regions should be in reading order, not DOM order
        assert_eq!(regions.len(), 3, "Expected 3 regions");
        
        assert_eq!(regions[0].id.as_deref(), Some("region_003"), 
            "First region should be region_003 (reading order index 0)");
        assert_eq!(regions[0].text.trim(), "First region text");
        
        assert_eq!(regions[1].id.as_deref(), Some("region_001"),
            "Second region should be region_001 (reading order index 1)");
        assert_eq!(regions[1].text.trim(), "Second region text");
        
        assert_eq!(regions[2].id.as_deref(), Some("region_002"),
            "Third region should be region_002 (reading order index 2)");
        assert_eq!(regions[2].text.trim(), "Third region text");
    }

    #[test]
    fn test_reading_order_with_missing_regions() {
        // Test that regions without reading order indices are placed after ordered ones
        let test_file = PathBuf::from("tests/resources/groundtruth/page/reading_order_with_missing_regions.xml");
        
        if !test_file.exists() {
            eprintln!("Skipping test: test file not found");
            return;
        }

        let content = std::fs::read_to_string(&test_file).unwrap();
        let doc = Document::parse(&content).unwrap();
        let (_text, regions) = parse_page_document(&doc).unwrap();

        // Assert - region_002 (with reading order) should come first
        assert_eq!(regions.len(), 2, "Expected 2 regions");
        
        assert_eq!(regions[0].id.as_deref(), Some("region_002"),
            "First region should be region_002 (has reading order)");
        assert_eq!(regions[0].text.trim(), "Has reading order");
        
        assert_eq!(regions[1].id.as_deref(), Some("region_001"),
            "Second region should be region_001 (no reading order)");
        assert_eq!(regions[1].text.trim(), "No reading order");
    }

    #[test]
    fn test_no_reading_order_element() {
        // Test that regions work correctly when no ReadingOrder element exists
        let test_file = PathBuf::from("tests/resources/groundtruth/page/no_reading_order_element.xml");
        
        if !test_file.exists() {
            eprintln!("Skipping test: test file not found");
            return;
        }

        let content = std::fs::read_to_string(&test_file).unwrap();
        let doc = Document::parse(&content).unwrap();
        let (_text, regions) = parse_page_document(&doc).unwrap();

        // Assert - without reading order, should maintain DOM order
        assert_eq!(regions.len(), 2, "Expected 2 regions");
        
        assert_eq!(regions[0].id.as_deref(), Some("region_001"),
            "First region should be region_001 (DOM order)");
        assert_eq!(regions[0].text.trim(), "First in DOM");
        
        assert_eq!(regions[1].id.as_deref(), Some("region_002"),
            "Second region should be region_002 (DOM order)");
        assert_eq!(regions[1].text.trim(), "Second in DOM");
    }

    #[test]
    fn test_page_with_bounding_boxes() {
        let xml = r#"<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page>
        <TextRegion id="r1">
            <Coords points="100,200 600,200 600,300 100,300"/>
            <TextLine id="l1">
                <Coords points="100,200 600,200 600,250 100,250"/>
                <TextEquiv><Unicode>Test</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"#;

        let doc = Document::parse(xml).unwrap();
        let (_, regions) = parse_page_document(&doc).unwrap();
        
        assert!(regions[0].bounding_box.is_some());
        let bbox = regions[0].bounding_box.as_ref().unwrap();
        
        assert_eq!(bbox.min_x, 100.0);
        assert_eq!(bbox.max_x, 600.0);
        assert_eq!(bbox.min_y, 200.0);
        assert_eq!(bbox.max_y, 300.0);
    }

    #[test]
    fn test_page_empty_unicode() {
        let xml = r#"<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page>
        <TextRegion id="r1">
            <Coords points="0,0 100,0 100,50 0,50"/>
            <TextLine id="l1">
                <Coords points="0,0 100,0 100,50 0,50"/>
                <TextEquiv><Unicode></Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_page_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].lines.len(), 1);
        assert_eq!(regions[0].lines[0].text, "");
        assert_eq!(text.trim(), "");
    }

    #[test]
    fn test_page_no_coords() {
        let xml = r#"<?xml version="1.0"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
    <Page>
        <TextRegion id="r1">
            <TextLine id="l1">
                <TextEquiv><Unicode>No coordinates</Unicode></TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_page_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 1);
        assert!(regions[0].bounding_box.is_none());
        assert!(text.contains("No coordinates"));
    }
}

