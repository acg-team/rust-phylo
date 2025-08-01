use std::error::Error;
use std::fmt::{self, Debug};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use anyhow::bail;
use bio::io::fasta::{Reader, Record, Writer};
use log::info;

use crate::alphabets::{protein_alphabet, GAP, POSSIBLE_GAPS};
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
/// All sequences are converted to uppercase.
///
/// # Arguments
/// * `path` - Path to the fasta file.
///
/// # Example
/// ```
/// use phylo::io::read_sequences;
/// # fn main() -> std::result::Result<(), anyhow::Error> {
/// let records = read_sequences("./examples/data/sequences_DNA_small.fasta")?;
/// assert_eq!(records.len(), 4);
/// for rec in records {
///    assert_eq!(rec.seq().len(), 8);
///    assert_eq!(rec.seq(), rec.seq().to_ascii_uppercase());
/// }
/// # Ok(()) }
/// ```
pub fn read_sequences(path: impl AsRef<Path> + Debug) -> Result<Vec<Record>> {
    info!("Reading sequences from file {}", path.as_ref().display());
    let reader = Reader::from_file(path)?;
    let mut sequences = Vec::new();

    for result in reader.records() {
        let rec = result?;
        if let Err(e) = rec.check() {
            bail!(DataError {
                message: e.to_string()
            });
        }
        let seq: Vec<u8> = rec
            .seq()
            .to_ascii_uppercase()
            .iter()
            .map(|c| if POSSIBLE_GAPS.contains(c) { GAP } else { *c })
            .collect();

        if !protein_alphabet().is_word(&seq) {
            bail!(DataError {
                message: format!(
                    "Invalid genetic sequence encountered: {}",
                    String::from_utf8(seq).unwrap()
                )
            });
        }

        sequences.push(Record::with_attrs(rec.id(), rec.desc(), &seq));
    }
    if sequences.is_empty() {
        bail!(DataError {
            message: String::from("No sequences found in file")
        });
    }

    info!("Read sequences successfully");
    Ok(sequences)
}

/// Writes fasta sequences to the given file path. Will return an error if the file already exists.
///
/// # Arguments
/// * `sequences` - Vector of fasta records.
/// * `path` - Path to the fasta file.
///
/// # TODO:
/// * Allow overwriting files if requested.
///
/// # Example
/// ```
/// # use std::io::Read;
///
/// use std::fs::{File, remove_file};
///
/// use bio::io::fasta::Record;
/// use phylo::io::write_sequences_to_file;
/// # fn main() -> std::result::Result<(), anyhow::Error> {
/// let sequences = vec![
///    Record::with_attrs("seq1", None, b"ATGC"),
///    Record::with_attrs("seq2", None, b"CGTA"),
/// ];
/// let output_path = "./examples/data/doctest_tmp_output.fasta";
/// write_sequences_to_file(&sequences, output_path)?;
/// # let mut file_content = String::new();
/// # File::open(output_path)?
/// #   .read_to_string(&mut file_content)?;
/// # let expected_output = ">seq1\nATGC\n>seq2\nCGTA\n";
/// # assert_eq!(file_content, expected_output);
/// # assert!(remove_file(output_path).is_ok());
/// # Ok(()) }
/// ```
pub fn write_sequences_to_file(sequences: &[Record], path: impl AsRef<Path>) -> Result<()> {
    info!("Writing sequences/MSA to file {}", path.as_ref().display());
    if path.as_ref().exists() {
        bail!(DataError {
            message: String::from("File already exists")
        });
    }
    let mut writer = Writer::to_file(path)?;
    for rec in sequences {
        writer.write_record(rec)?;
    }
    info!("Finished writing successfully");
    Ok(())
}

/// Reads newick trees from a file, returning a vector of trees.
///
/// Will read both rooted and unrooted trees, but unrooted trees will be converted to rooted
/// using zero length branches at the trifurcation.
/// For example, the unrooted tree "((A:1,B:2):1,(D:1,E:2):1,C:4);" will be converted to the rooted
/// tree "(((A:1,B:2):1,(D:1,E:2):1):0,C:4):0;".
///
/// # Arguments
/// * `path` - Path to the newick file.
///
/// # Example
/// ```
/// use phylo::io::read_newick_from_file;
/// # fn main() -> std::result::Result<(), anyhow::Error> {
/// let trees = read_newick_from_file("./examples/data/tree.newick")?;
/// assert_eq!(trees.len(), 1);
/// assert_eq!(trees[0].leaves().len(), 4);
/// # Ok(()) }
/// ```
pub fn read_newick_from_file(path: impl AsRef<Path>) -> Result<Vec<Tree>> {
    info!("Reading newick trees from file {}", path.as_ref().display());
    let newick = fs::read_to_string(path)?;
    info!("Read file successfully");
    tree_parser::from_newick(&newick)
}

/// Writes newick trees to the given file path. Will return an error if the file already exists.
///
/// # Arguments
/// * `trees` - Vector of newick trees.
/// * `path` - Path to the newick file.
///
/// # Example
/// ```
/// # use std::fs::{File, remove_file};
/// # use std::io::Read;
///
/// use phylo::tree::tree_parser::from_newick;
/// use phylo::tree::Tree;
/// use phylo::io::write_newick_to_file;
///
/// # fn main() -> std::result::Result<(), anyhow::Error> {
/// let output_path = "./examples/data/doctest_tmp_output.newick";
/// let trees = from_newick("((A:1.0,B:2.0):1,(D:1.0,E:2.0):1):0.0;")?;
/// write_newick_to_file(&trees, output_path)?;
/// # let mut file_content = String::new();
/// # File::open(output_path)?.read_to_string(&mut file_content)?;
/// # assert_eq!(file_content.trim(), "(((A:1,B:2):1,(D:1,E:2):1):0);");
/// # assert!(remove_file(output_path).is_ok());
/// # Ok(()) }
/// ```
pub fn write_newick_to_file(trees: &[Tree], path: impl AsRef<Path>) -> Result<()> {
    info!("Writing newick trees to file {}", path.as_ref().display());
    if path.as_ref().exists() {
        bail!(DataError {
            message: String::from("File already exists")
        });
    }
    let mut writer = File::create(path)?;
    for tree in trees {
        writer.write_all(tree.to_newick().as_bytes())?;
        writer.write_all(b"\n")?;
    }
    info!("Finished writing successfully");
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
