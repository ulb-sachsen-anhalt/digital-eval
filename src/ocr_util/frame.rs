use anyhow::Result;
use std::fs;
use std::path::Path;
use roxmltree::{Document, Node};
use std::io::Write;

/// Filter an ALTO file to only include content within specified area
pub fn filter_frame(input_path: &Path, output_path: &Path, points_str: &str) -> Result<()> {
    // Parse the points
    let coords = parse_points(points_str)?;
    
    if coords.is_empty() {
        anyhow::bail!("No valid coordinates provided");
    }

    // Handle simple rectangle case (2 points)
    let rect = if coords.len() == 2 {
        // Two points: top-left and bottom-right
        Some((coords[0].0, coords[0].1, coords[1].0, coords[1].1))
    } else if coords.len() == 4 && is_rectangle(&coords) {
        // Four points forming a rectangle
        let min_x = coords.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let min_y = coords.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let max_x = coords.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let max_y = coords.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
        Some((min_x, min_y, max_x, max_y))
    } else {
        // Complex polygon - would need more sophisticated filtering
        let min_x = coords.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let min_y = coords.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let max_x = coords.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let max_y = coords.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
        Some((min_x, min_y, max_x, max_y))
    };

    let (min_x, min_y, max_x, max_y) = rect.unwrap();

    // Read and parse input ALTO file
    let content = fs::read_to_string(input_path)?;
    let doc = Document::parse(&content)?;

    // Filter and write output
    let filtered_xml = filter_alto_content(&doc, min_x, min_y, max_x, max_y)?;
    
    let mut output_file = fs::File::create(output_path)?;
    output_file.write_all(filtered_xml.as_bytes())?;

    Ok(())
}

/// Parse points from string format "x1,y1 x2,y2 ..."
fn parse_points(points_str: &str) -> Result<Vec<(f64, f64)>> {
    let mut coords = Vec::new();

    for point in points_str.split_whitespace() {
        let parts: Vec<&str> = point.split(',').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid point format: {}. Expected 'x,y'", point);
        }

        let x = parts[0].trim().parse::<f64>()?;
        let y = parts[1].trim().parse::<f64>()?;
        coords.push((x, y));
    }

    Ok(coords)
}

/// Check if points form a rectangle
fn is_rectangle(coords: &[(f64, f64)]) -> bool {
    if coords.len() != 4 {
        return false;
    }

    // Check if we have exactly 2 unique x values and 2 unique y values
    let mut x_vals: Vec<f64> = coords.iter().map(|(x, _)| *x).collect();
    let mut y_vals: Vec<f64> = coords.iter().map(|(_, y)| *y).collect();
    
    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let x_unique = x_vals.windows(2).filter(|w| (w[0] - w[1]).abs() > 0.01).count() + 1;
    let y_unique = y_vals.windows(2).filter(|w| (w[0] - w[1]).abs() > 0.01).count() + 1;

    x_unique == 2 && y_unique == 2
}

/// Filter ALTO content based on bounding box
fn filter_alto_content(doc: &Document, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Result<String> {
    // This is a simplified version - a full implementation would properly rebuild the XML tree
    // For now, we'll create a basic filtered ALTO structure
    
    let mut output = String::from(r#"<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
    <Description>
        <MeasurementUnit>pixel</MeasurementUnit>
    </Description>
    <Layout>
        <Page>
            <PrintSpace>
"#);

    // Find and filter TextBlocks
    for node in doc.descendants() {
        if node.has_tag_name("TextBlock") {
            if let Some(filtered_block) = filter_text_block(&node, min_x, min_y, max_x, max_y) {
                output.push_str(&filtered_block);
            }
        }
    }

    output.push_str(r#"            </PrintSpace>
        </Page>
    </Layout>
</alto>"#);

    Ok(output)
}

/// Filter a TextBlock based on bounding box
fn filter_text_block(node: &Node, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Option<String> {
    // Check if TextBlock intersects with the filter area
    let hpos = node.attribute("HPOS")?.parse::<f64>().ok()?;
    let vpos = node.attribute("VPOS")?.parse::<f64>().ok()?;
    let width = node.attribute("WIDTH")?.parse::<f64>().ok()?;
    let height = node.attribute("HEIGHT")?.parse::<f64>().ok()?;

    let block_max_x = hpos + width;
    let block_max_y = vpos + height;

    // Check intersection
    if hpos > max_x || block_max_x < min_x || vpos > max_y || block_max_y < min_y {
        return None; // No intersection
    }

    // Build filtered TextBlock XML (simplified)
    let id = node.attribute("ID").unwrap_or("block");
    let mut result = format!(
        r#"                <TextBlock ID="{}" HPOS="{}" VPOS="{}" WIDTH="{}" HEIGHT="{}">
"#,
        id, hpos, vpos, width, height
    );

    // Add TextLines
    for child in node.descendants() {
        if child.has_tag_name("TextLine") {
            if let Some(line) = filter_text_line(&child, min_x, min_y, max_x, max_y) {
                result.push_str(&line);
            }
        }
    }

    result.push_str("                </TextBlock>\n");
    Some(result)
}

/// Filter a TextLine based on bounding box
fn filter_text_line(node: &Node, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Option<String> {
    let hpos = node.attribute("HPOS")?.parse::<f64>().ok()?;
    let vpos = node.attribute("VPOS")?.parse::<f64>().ok()?;
    let width = node.attribute("WIDTH")?.parse::<f64>().ok()?;
    let height = node.attribute("HEIGHT")?.parse::<f64>().ok()?;

    let line_max_x = hpos + width;
    let line_max_y = vpos + height;

    // Check intersection
    if hpos > max_x || line_max_x < min_x || vpos > max_y || line_max_y < min_y {
        return None;
    }

    let id = node.attribute("ID").unwrap_or("line");
    let mut result = format!(
        r#"                    <TextLine ID="{}" HPOS="{}" VPOS="{}" WIDTH="{}" HEIGHT="{}">
"#,
        id, hpos, vpos, width, height
    );

    // Add Strings (words)
    for child in node.descendants() {
        if child.has_tag_name("String") {
            if let Some(content) = child.attribute("CONTENT") {
                let w_hpos = child.attribute("HPOS").and_then(|s| s.parse::<f64>().ok()).unwrap_or(hpos);
                let w_vpos = child.attribute("VPOS").and_then(|s| s.parse::<f64>().ok()).unwrap_or(vpos);
                let w_width = child.attribute("WIDTH").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                let w_height = child.attribute("HEIGHT").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                
                result.push_str(&format!(
                    r#"                        <String CONTENT="{}" HPOS="{}" VPOS="{}" WIDTH="{}" HEIGHT="{}"/>
"#,
                    content, w_hpos, w_vpos, w_width, w_height
                ));
            }
        }
    }

    result.push_str("                    </TextLine>\n");
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_points() {
        let points = "0,0 100,200";
        let coords = parse_points(points).unwrap();
        assert_eq!(coords.len(), 2);
        assert_eq!(coords[0], (0.0, 0.0));
        assert_eq!(coords[1], (100.0, 200.0));
    }

    #[test]
    fn test_is_rectangle() {
        let rect = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        assert!(is_rectangle(&rect));

        let not_rect = vec![(0.0, 0.0), (50.0, 50.0), (100.0, 0.0)];
        assert!(!is_rectangle(&not_rect));
    }
}
