use geo::{Point, Polygon, Rect, Contains};
use anyhow::Result;

/// Represents a 2D point with coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coordinate {
    pub x: f64,
    pub y: f64,
}

impl Coordinate {
    pub fn new(x: f64, y: f64) -> Self {
        Coordinate { x, y }
    }

    pub fn to_point(&self) -> Point<f64> {
        Point::new(self.x, self.y)
    }
}

/// Represents a bounding box
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BoundingBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        BoundingBox { min_x, min_y, max_x, max_y }
    }

    pub fn from_points(points: &[Coordinate]) -> Result<Self> {
        if points.is_empty() {
            anyhow::bail!("Cannot create bounding box from empty points");
        }

        let min_x = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let min_y = points.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let max_y = points.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

        Ok(BoundingBox { min_x, min_y, max_x, max_y })
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub fn contains_point(&self, point: &Coordinate) -> bool {
        point.x >= self.min_x 
            && point.x <= self.max_x 
            && point.y >= self.min_y 
            && point.y <= self.max_y
    }

    pub fn to_rect(&self) -> Rect<f64> {
        Rect::new(
            geo::Coord { x: self.min_x, y: self.min_y },
            geo::Coord { x: self.max_x, y: self.max_y },
        )
    }
}

/// Get bounding box from a polygon
pub fn get_bounding_box(coordinates: &[Coordinate]) -> Result<BoundingBox> {
    BoundingBox::from_points(coordinates)
}

/// Parse polygon from string (e.g., "0,0 100,0 100,100 0,100")
pub fn parse_polygon_string(polygon_str: &str) -> Result<Vec<Coordinate>> {
    let mut coordinates = Vec::new();

    for point_str in polygon_str.split_whitespace() {
        let parts: Vec<&str> = point_str.split(',').collect();
        if parts.len() != 2 {
            anyhow::bail!("Invalid point format: {}", point_str);
        }

        let x = parts[0].parse::<f64>()
            .map_err(|e| anyhow::anyhow!("Invalid x coordinate: {}", e))?;
        let y = parts[1].parse::<f64>()
            .map_err(|e| anyhow::anyhow!("Invalid y coordinate: {}", e))?;

        coordinates.push(Coordinate::new(x, y));
    }

    if coordinates.len() < 3 {
        anyhow::bail!("Polygon must have at least 3 points");
    }

    Ok(coordinates)
}

/// Check if a point is inside a polygon
pub fn point_in_polygon(point: &Coordinate, polygon: &[Coordinate]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let geo_point = point.to_point();
    let geo_polygon = create_geo_polygon(polygon);

    geo_polygon.contains(&geo_point)
}

/// Create a geo::Polygon from coordinates
fn create_geo_polygon(coords: &[Coordinate]) -> Polygon<f64> {
    let points: Vec<geo::Coord<f64>> = coords
        .iter()
        .map(|c| geo::Coord { x: c.x, y: c.y })
        .collect();

    Polygon::new(geo::LineString::from(points), vec![])
}

/// Calculate intersection area between two bounding boxes
pub fn intersection_area(box1: &BoundingBox, box2: &BoundingBox) -> f64 {
    let x_overlap = f64::max(0.0, f64::min(box1.max_x, box2.max_x) - f64::max(box1.min_x, box2.min_x));
    let y_overlap = f64::max(0.0, f64::min(box1.max_y, box2.max_y) - f64::max(box1.min_y, box2.min_y));
    
    x_overlap * y_overlap
}

/// Calculate union area between two bounding boxes
pub fn union_area(box1: &BoundingBox, box2: &BoundingBox) -> f64 {
    let area1 = box1.area();
    let area2 = box2.area();
    let intersection = intersection_area(box1, box2);
    
    area1 + area2 - intersection
}

/// Calculate Intersection over Union (IoU) for two bounding boxes
pub fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f64 {
    let intersection = intersection_area(box1, box2);
    let union = union_area(box1, box2);
    
    if union == 0.0 {
        return 0.0;
    }
    
    intersection / union
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinate_creation() {
        let coord = Coordinate::new(10.0, 20.0);
        assert_eq!(coord.x, 10.0);
        assert_eq!(coord.y, 20.0);
    }

    #[test]
    fn test_bounding_box_creation() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        assert_eq!(bbox.width(), 100.0);
        assert_eq!(bbox.height(), 100.0);
        assert_eq!(bbox.area(), 10000.0);
    }

    #[test]
    fn test_bounding_box_from_points() {
        let points = vec![
            Coordinate::new(10.0, 20.0),
            Coordinate::new(50.0, 60.0),
            Coordinate::new(30.0, 40.0),
        ];
        let bbox = BoundingBox::from_points(&points).unwrap();
        assert_eq!(bbox.min_x, 10.0);
        assert_eq!(bbox.min_y, 20.0);
        assert_eq!(bbox.max_x, 50.0);
        assert_eq!(bbox.max_y, 60.0);
    }

    #[test]
    fn test_contains_point() {
        let bbox = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        assert!(bbox.contains_point(&Coordinate::new(50.0, 50.0)));
        assert!(!bbox.contains_point(&Coordinate::new(150.0, 50.0)));
    }

    #[test]
    fn test_parse_polygon_string() {
        let polygon_str = "0,0 100,0 100,100 0,100";
        let coords = parse_polygon_string(polygon_str).unwrap();
        assert_eq!(coords.len(), 4);
        assert_eq!(coords[0], Coordinate::new(0.0, 0.0));
        assert_eq!(coords[2], Coordinate::new(100.0, 100.0));
    }

    #[test]
    fn test_intersection_area() {
        let box1 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let box2 = BoundingBox::new(50.0, 50.0, 150.0, 150.0);
        let intersection = intersection_area(&box1, &box2);
        assert_eq!(intersection, 2500.0); // 50x50
    }

    #[test]
    fn test_calculate_iou() {
        let box1 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let box2 = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let iou = calculate_iou(&box1, &box2);
        assert_eq!(iou, 1.0); // Perfect overlap
    }
}
