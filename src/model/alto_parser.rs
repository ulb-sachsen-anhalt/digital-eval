use anyhow::Result;
use roxmltree::{Document, Node};

use super::digital_object::{Region, TextLine, Word};
use crate::geometry::{BoundingBox, Coordinate};

/// Parse an ALTO XML document
pub fn parse_alto_document(doc: &Document) -> Result<(String, Vec<Region>)> {
    let mut full_text = String::new();
    let mut regions = Vec::new();

    // Find all TextBlock elements
    for node in doc.descendants() {
        if node.has_tag_name("TextBlock") {
            if let Ok(region) = parse_text_block(&node) {
                full_text.push_str(&region.text);
                full_text.push('\n');
                regions.push(region);
            }
        }
    }

    Ok((full_text.trim().to_string(), regions))
}

/// Parse a TextBlock element
fn parse_text_block(node: &Node) -> Result<Region> {
    let id = node.attribute("ID").map(|s| s.to_string());
    let bounding_box = parse_bounding_box(node);

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
    let id = node.attribute("ID").map(|s| s.to_string());
    let bounding_box = parse_bounding_box(node);

    let mut words = Vec::new();
    let mut line_text = String::new();

    // Find all String elements (words)
    for child in node.descendants() {
        if child.has_tag_name("String") {
            if let Some(content) = child.attribute("CONTENT") {
                if !line_text.is_empty() {
                    line_text.push(' ');
                }
                line_text.push_str(content);

                let word = Word {
                    text: content.to_string(),
                    confidence: child.attribute("WC")
                        .and_then(|s| s.parse::<f64>().ok()),
                    bounding_box: parse_bounding_box(&child),
                };
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

/// Parse bounding box attributes from a node
fn parse_bounding_box(node: &Node) -> Option<BoundingBox> {
    let hpos = node.attribute("HPOS")?.parse::<f64>().ok()?;
    let vpos = node.attribute("VPOS")?.parse::<f64>().ok()?;
    let width = node.attribute("WIDTH")?.parse::<f64>().ok()?;
    let height = node.attribute("HEIGHT")?.parse::<f64>().ok()?;

    Some(BoundingBox::new(hpos, vpos, hpos + width, vpos + height))
}

/// Parse polygon coordinates from ALTO
#[allow(dead_code)]
pub fn parse_alto_polygon(points_str: &str) -> Result<Vec<Coordinate>> {
    let mut coordinates = Vec::new();
    let parts: Vec<&str> = points_str.split_whitespace().collect();

    for i in (0..parts.len()).step_by(2) {
        if i + 1 >= parts.len() {
            break;
        }

        let x = parts[i].parse::<f64>()?;
        let y = parts[i + 1].parse::<f64>()?;
        coordinates.push(Coordinate::new(x, y));
    }

    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_parse_alto_polygon() {
        let points = "100 200 300 200 300 400 100 400";
        let coords = parse_alto_polygon(points).unwrap();
        assert_eq!(coords.len(), 4);
        assert_eq!(coords[0].x, 100.0);
        assert_eq!(coords[0].y, 200.0);
        assert_eq!(coords[3].x, 100.0);
        assert_eq!(coords[3].y, 400.0);
    }

    #[test]
    fn test_parse_alto_polygon_odd_count() {
        let points = "100 200 300";
        let coords = parse_alto_polygon(points).unwrap();
        // Should only parse complete pairs
        assert_eq!(coords.len(), 1);
    }

    #[test]
    fn test_parse_simple_alto() {
        let xml = r#"<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
    <Layout>
        <Page>
            <PrintSpace>
                <TextBlock ID="TB1" HPOS="100" VPOS="200" WIDTH="500" HEIGHT="100">
                    <TextLine ID="TL1" HPOS="100" VPOS="200" WIDTH="500" HEIGHT="50">
                        <String ID="S1" CONTENT="Hello" HPOS="100" VPOS="200" WIDTH="100" HEIGHT="50" WC="0.95"/>
                        <String ID="S2" CONTENT="World" HPOS="210" VPOS="200" WIDTH="100" HEIGHT="50" WC="0.98"/>
                    </TextLine>
                </TextBlock>
            </PrintSpace>
        </Page>
    </Layout>
</alto>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_alto_document(&doc).unwrap();
        
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!regions.is_empty());
        assert_eq!(regions.len(), 1);
        
        // Check region ID
        assert_eq!(regions[0].id.as_deref(), Some("TB1"));
        
        // Check lines
        assert_eq!(regions[0].lines.len(), 1);
        assert_eq!(regions[0].lines[0].id.as_deref(), Some("TL1"));
        
        // Check words
        assert_eq!(regions[0].lines[0].words.len(), 2);
        assert_eq!(regions[0].lines[0].words[0].text, "Hello");
        assert_eq!(regions[0].lines[0].words[1].text, "World");
        
        // Check confidence
        assert_eq!(regions[0].lines[0].words[0].confidence, Some(0.95));
    }

    #[test]
    fn test_parse_alto_multiple_textblocks() {
        let xml = r#"<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
    <Layout>
        <Page>
            <PrintSpace>
                <TextBlock ID="block_1" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="50">
                    <TextLine ID="line_1" HPOS="0" VPOS="0" WIDTH="100" HEIGHT="50">
                        <String CONTENT="First" HPOS="0" VPOS="0" WIDTH="50" HEIGHT="50"/>
                    </TextLine>
                </TextBlock>
                <TextBlock ID="block_2" HPOS="0" VPOS="60" WIDTH="100" HEIGHT="50">
                    <TextLine ID="line_2" HPOS="0" VPOS="60" WIDTH="100" HEIGHT="50">
                        <String CONTENT="Second" HPOS="0" VPOS="60" WIDTH="60" HEIGHT="50"/>
                    </TextLine>
                </TextBlock>
            </PrintSpace>
        </Page>
    </Layout>
</alto>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_alto_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 2);
        assert!(text.contains("First"));
        assert!(text.contains("Second"));
    }

    #[test]
    fn test_parse_alto_with_bounding_boxes() {
        let xml = r#"<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
    <Layout>
        <Page>
            <PrintSpace>
                <TextBlock ID="TB1" HPOS="100" VPOS="200" WIDTH="500" HEIGHT="100">
                    <TextLine ID="TL1" HPOS="100" VPOS="200" WIDTH="500" HEIGHT="50">
                        <String CONTENT="Test" HPOS="100" VPOS="200" WIDTH="100" HEIGHT="50"/>
                    </TextLine>
                </TextBlock>
            </PrintSpace>
        </Page>
    </Layout>
</alto>"#;

        let doc = Document::parse(xml).unwrap();
        let (_, regions) = parse_alto_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 1);
        
        // Check bounding box
        assert!(regions[0].bounding_box.is_some());
        let bbox = regions[0].bounding_box.as_ref().unwrap();
        assert_eq!(bbox.min_x, 100.0);
        assert_eq!(bbox.min_y, 200.0);
        assert_eq!(bbox.max_x, 600.0); // 100 + 500
        assert_eq!(bbox.max_y, 300.0); // 200 + 100
    }

    #[test]
    fn test_alto_multiple_lines_in_block() {
        let xml = r#"<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
    <Layout>
        <Page>
            <PrintSpace>
                <TextBlock ID="block_1" HPOS="0" VPOS="0" WIDTH="200" HEIGHT="100">
                    <TextLine ID="line_1" HPOS="0" VPOS="0" WIDTH="200" HEIGHT="40">
                        <String CONTENT="Line" HPOS="0" VPOS="0" WIDTH="50" HEIGHT="40"/>
                        <String CONTENT="One" HPOS="60" VPOS="0" WIDTH="40" HEIGHT="40"/>
                    </TextLine>
                    <TextLine ID="line_2" HPOS="0" VPOS="50" WIDTH="200" HEIGHT="40">
                        <String CONTENT="Line" HPOS="0" VPOS="50" WIDTH="50" HEIGHT="40"/>
                        <String CONTENT="Two" HPOS="60" VPOS="50" WIDTH="40" HEIGHT="40"/>
                    </TextLine>
                </TextBlock>
            </PrintSpace>
        </Page>
    </Layout>
</alto>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_alto_document(&doc).unwrap();
        
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].lines.len(), 2);
        
        assert_eq!(regions[0].lines[0].words.len(), 2);
        assert_eq!(regions[0].lines[1].words.len(), 2);
        
        assert!(text.contains("Line One"));
        assert!(text.contains("Line Two"));
    }

    #[test]
    fn test_alto_groundtruth_file() {
        let test_file = PathBuf::from("tests/resources/groundtruth/alto/1667522809_J_0073_0001_375x2050_2325x9550.xml");
        
        if test_file.exists() {
            let content = std::fs::read_to_string(&test_file).unwrap();
            let doc = Document::parse(&content).unwrap();
            let (text, regions) = parse_alto_document(&doc).unwrap();
            
            // Based on Python tests
            assert_eq!(regions.len(), 10);
            assert!(!text.is_empty());
            
            // Check IDs
            assert_eq!(regions[0].id.as_deref(), Some("block_27"));
            assert_eq!(regions[1].id.as_deref(), Some("block_28"));
            
            // Region 2 (index 1) should have 2 lines
            assert_eq!(regions[1].lines.len(), 2);
            
            // First line of first region should have 2 words
            assert_eq!(regions[0].lines[0].words.len(), 2);
        }
    }

    #[test]
    fn test_alto_empty_string_content() {
        let xml = r#"<?xml version="1.0"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
    <Layout>
        <Page>
            <PrintSpace>
                <TextBlock ID="TB1" HPOS="100" VPOS="200" WIDTH="500" HEIGHT="100">
                    <TextLine ID="TL1" HPOS="100" VPOS="200" WIDTH="500" HEIGHT="50">
                        <String ID="S1" HPOS="100" VPOS="200" WIDTH="100" HEIGHT="50"/>
                    </TextLine>
                </TextBlock>
            </PrintSpace>
        </Page>
    </Layout>
</alto>"#;

        let doc = Document::parse(xml).unwrap();
        let (text, regions) = parse_alto_document(&doc).unwrap();
        
        // Should handle missing CONTENT attribute gracefully
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].lines[0].words.len(), 0); // No words added without CONTENT
    }
}

