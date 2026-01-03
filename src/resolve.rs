use anyhow::Result;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Evaluation entry containing candidate and groundtruth paths
#[derive(Debug, Clone)]
pub struct EvalEntry {
    pub path_candidate: PathBuf,
    pub candidate_root_domain: Option<PathBuf>,
    pub domain_directories: Vec<String>,
    pub path_groundtruth: Option<PathBuf>,
    pub gt_type: String,
    pub metrics: Vec<f64>,
}

impl EvalEntry {
    pub fn new(path: PathBuf, candidate_root: Option<PathBuf>) -> Self {
        EvalEntry {
            path_candidate: path,
            candidate_root_domain: candidate_root,
            domain_directories: Vec::new(),
            path_groundtruth: None,
            gt_type: "n.a.".to_string(),
            metrics: Vec::new(),
        }
    }

    pub fn set_groundtruth(&mut self, gt_path: PathBuf) {
        self.path_groundtruth = Some(gt_path);
    }

    pub fn has_groundtruth(&self) -> bool {
        self.path_groundtruth.is_some()
    }

    pub fn align_domains(&mut self) {
        if self.path_groundtruth.is_none() || self.candidate_root_domain.is_none() {
            return;
        }

        let candidate_root = self.candidate_root_domain.as_ref().unwrap();
        let candidate_name = candidate_root.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        let mut dirs = Vec::new();
        let mut current = self.path_candidate.parent();

        while let Some(parent) = current {
            if let Some(name) = parent.file_name().and_then(|s| s.to_str()) {
                if name == candidate_name {
                    break;
                }
                if name != "GT-PAGE" {
                    dirs.push(name.to_string());
                }
            }
            current = parent.parent();
        }

        dirs.reverse();
        self.domain_directories = dirs;
    }
}

/// Gather all candidate files from a directory
pub fn gather_candidates(start_path: &Path) -> Result<Vec<EvalEntry>> {
    gather(start_path, ".xml")
}

/// Gather all files with a specific extension from start_path
pub fn gather(start_path: &Path, file_ext: &str) -> Result<Vec<EvalEntry>> {
    let mut candidates = Vec::new();

    if !start_path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", start_path.display());
    }

    for entry in WalkDir::new(start_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                if filename.ends_with(file_ext) {
                    let eval_entry = EvalEntry::new(
                        path.to_path_buf(),
                        Some(start_path.to_path_buf()),
                    );
                    candidates.push(eval_entry);
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.path_candidate.cmp(&b.path_candidate));
    Ok(candidates)
}

/// Find corresponding groundtruth file for a candidate
pub fn find_groundtruth(eval_entry: &EvalEntry, gt_domain_root: &Path) -> Option<PathBuf> {
    let candidate_stem = eval_entry.path_candidate
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let mut gt_files = Vec::new();

    for entry in WalkDir::new(gt_domain_root)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                if name_approved(filename, candidate_stem) {
                    gt_files.push(path.to_path_buf());
                }
            }
        }
    }

    gt_files.into_iter().next()
}

/// Check if filename matches the expected candidate name
fn name_approved(fname: &str, estm_name: &str) -> bool {
    let suffix_ok = fname.ends_with(".gt.xml") 
        || fname.ends_with("gt.txt") 
        || fname.ends_with(".xml");
    
    fname.starts_with(estm_name) && suffix_ok
}

/// Subtract one path from another to get the relative difference
pub fn subtract_paths(absolute_path: &Path, base_path: &Path) -> Result<PathBuf> {
    absolute_path
        .strip_prefix(base_path)
        .map(|p| p.to_path_buf())
        .map_err(|e| anyhow::anyhow!("Path '{}' is not relative to '{}': {}", 
                                     absolute_path.display(), 
                                     base_path.display(), 
                                     e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_eval_entry_creation() {
        let path = PathBuf::from("/test/path/file.xml");
        let root = Some(PathBuf::from("/test/path"));
        let entry = EvalEntry::new(path.clone(), root);
        
        assert_eq!(entry.path_candidate, path);
        assert!(!entry.has_groundtruth());
    }

    #[test]
    fn test_name_approved() {
        assert!(name_approved("file.gt.xml", "file"));
        assert!(name_approved("file.xml", "file"));
        assert!(name_approved("file123.gt.xml", "file123"));
        assert!(!name_approved("other.xml", "file"));
    }

    #[test]
    fn test_subtract_paths() {
        let abs_path = Path::new("/home/user/project/src/file.rs");
        let base_path = Path::new("/home/user/project");
        let result = subtract_paths(abs_path, base_path).unwrap();
        assert_eq!(result, PathBuf::from("src/file.rs"));
    }

    #[test]
    fn test_gather_empty_dir() {
        let temp_dir = TempDir::new().unwrap();
        let result = gather(temp_dir.path(), ".xml").unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_gather_with_files() {
        let temp_dir = TempDir::new().unwrap();
        let file1 = temp_dir.path().join("test1.xml");
        let file2 = temp_dir.path().join("test2.xml");
        let file3 = temp_dir.path().join("test.txt");
        
        fs::write(&file1, "content").unwrap();
        fs::write(&file2, "content").unwrap();
        fs::write(&file3, "content").unwrap();
        
        let result = gather(temp_dir.path(), ".xml").unwrap();
        assert_eq!(result.len(), 2);
    }
}
