use std::{error::Error, fmt, fs, path::PathBuf};

use anyhow::bail;
use bio::{
    alphabets,
    io::fasta::{Reader, Record, Writer},
};
use log::info;

use crate::tree::{tree_parser, Tree};
use crate::Result;

pub(crate) struct DataError {
    pub(crate) message: String,
}
impl fmt::Debug for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl Error for DataError {}

/// Reads sequences from a fasta file, returning a vector of fasta records.
///
/// # Arguments
/// * `path` - Path to the fasta file.
///
/// # Example
/// ```
/// use phylo::io::read_sequences_from_file;
/// use std::path::PathBuf;
/// let records = read_sequences_from_file(PathBuf::from("./data/sequences_DNA_small.fasta")).unwrap();
/// assert_eq!(records.len(), 4);
/// for rec in records {
///     assert_eq!(rec.seq().len(), 7);
/// }
/// ```
pub fn read_sequences_from_file(path: PathBuf) -> Result<Vec<Record>> {
    info!("Reading sequences from file {}.", path.display());
    let reader = Reader::from_file(path)?;
    let mut sequences = Vec::new();
    let mut alphabet = alphabets::protein::iupac_alphabet();
    alphabet.insert(b'-');
    for result in reader.records() {
        let rec = result?;
        if let Err(e) = rec.check() {
            bail!(DataError {
                message: e.to_string()
            });
        }
        if !alphabet.is_word(rec.seq()) {
            bail!(DataError {
                message: String::from("Invalid genetic sequences")
            });
        }
        sequences.push(rec);
    }
    info!("Read sequences successfully.");
    Ok(sequences)
}

/// Writes fasta sequences to the given file path. Will return an error if the file already exists.
///
/// # Arguments
/// * `sequences` - Vector of fasta records.
/// * `path` - Path to the fasta file.
///
/// # Example
/// ```
/// use std::path::PathBuf;
/// use std::fs::{File, remove_file};
/// use std::io::Read;
/// use std::time::{SystemTime, UNIX_EPOCH};
/// use bio::io::fasta::Record;
/// use phylo::io::write_sequences_to_file;
/// let sequences = vec![
///    Record::with_attrs("seq1", None, b"ATGC"),
///    Record::with_attrs("seq2", None, b"CGTA"),
/// ];
/// let output_path = PathBuf::from(format!(
///     "./data/doctest_tmp_output_{}.fasta",
///     SystemTime::now()
///         .duration_since(UNIX_EPOCH)
///         .unwrap()
///         .as_secs()
/// ));
/// write_sequences_to_file(&sequences, output_path.clone()).unwrap();
/// let mut file_content = String::new();
/// File::open(output_path.clone())
///    .unwrap()
///    .read_to_string(&mut file_content)
///    .unwrap();
/// let expected_output = ">seq1\nATGC\n>seq2\nCGTA\n";
/// assert_eq!(file_content, expected_output);
/// assert!(remove_file(output_path).is_ok());
/// ```
pub fn write_sequences_to_file(sequences: &[Record], path: PathBuf) -> Result<()> {
    info!("Writing sequences/MSA to file {}.", path.display());
    let mut writer = Writer::to_file(path)?;
    for rec in sequences {
        writer.write_record(rec)?;
    }
    info!("Finished writing successfully.");
    Ok(())
}

/// Reads rooted newick trees from a file, returning a vector of trees.
/// Currently can only process rooted trees, will return an error otherwise.
///
/// # Arguments
/// * `path` - Path to the newick file.
///
/// # Example
/// ```
/// use phylo::io::read_newick_from_file;
/// use std::path::PathBuf;
/// let trees = read_newick_from_file(PathBuf::from("./data/tree.newick")).unwrap();
/// assert_eq!(trees.len(), 1);
/// assert_eq!(trees[0].leaves.len(), 4);
/// assert!(read_newick_from_file(PathBuf::from("./data/tree_unrooted.newick")).is_err());
/// ```
pub fn read_newick_from_file(path: PathBuf) -> Result<Vec<Tree>> {
    info!("Reading rooted newick tree from file {}.", path.display());
    let newick = fs::read_to_string(path)?;
    info!("Read file successfully.");
    tree_parser::from_newick_string(&newick)
}

#[cfg(test)]
mod io_tests;
