use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::bail;
use bio::alphabets::Alphabet;
use bio::io::fasta::Record;
use log::{info, warn};
use nalgebra::{DMatrix, DVector};

use crate::io::{self, DataError};
use crate::sequences::{get_sequence_type, SequenceType};
use crate::substitution_models::dna_models::{DNA_GAP_SETS, DNA_SETS};
use crate::substitution_models::protein_models::{PROTEIN_GAP_SETS, PROTEIN_SETS};
use crate::tree::Tree;
use crate::Result;

/// Gap handling options. Ambiguous means that gaps are treated as unknown characters (X),
/// Proper means that the gaps are treated as a separate character.
pub enum GapHandling {
    Ambiguous,
    Proper,
}

/// The PhyloInfo struct contains all the information needed to run a phylogenetic analysis.
///
/// # TODO:
/// * Add getters/setters for the fields, make the fields private.
#[derive(Debug)]
pub struct PhyloInfo {
    /// Unaligned phylogenetic sequences.
    pub sequences: Vec<Record>,
    /// Multiple sequence alignment of the sequences, if they are aligned.
    pub msa: Option<Vec<Record>>,
    /// Phylogenetic tree.
    pub tree: Tree,
    /// Leaf sequences as probability vectors for characters in the alphabet.
    pub leaf_encoding: Vec<DMatrix<f64>>,
}

/// Converts the given sequences to uppercase and returns a new vector.
fn make_sequences_uppercase(sequences: &[Record]) -> Vec<Record> {
    sequences
        .iter()
        .map(|rec| Record::with_attrs(rec.id(), rec.desc(), &rec.seq().to_ascii_uppercase()))
        .collect()
}

impl PhyloInfo {
    /// Returns the empirical frequencies of the symbols in the sequences.
    /// The frequencies are calculated from the unaligned sequences.
    ///
    /// # Arguments
    /// * `alphabet` - Alphabet of the sequences.
    ///
    /// # Example
    /// ```
    /// # use bio::io::fasta::Record;
    /// use phylo::phylo_info::phyloinfo_from_sequences_tree;
    /// use phylo::phylo_info::GapHandling;
    /// use phylo::sequences::dna_alphabet;
    /// use phylo::tree::tree_parser::from_newick_string;
    /// let sequences = vec![
    ///     Record::with_attrs("A", None, b"AAAAA"),
    ///     Record::with_attrs("B", None, b"CCCCC"),
    ///     Record::with_attrs("C", None, b"GGGGG"),
    ///     Record::with_attrs("D", None, b"TTTTT"),
    /// ];
    /// let tree = from_newick_string("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);").unwrap().pop().unwrap();
    /// let info = phyloinfo_from_sequences_tree(&sequences, tree, &GapHandling::Ambiguous).unwrap();
    /// let freqs = info.get_counts(&dna_alphabet());
    /// assert_eq!(freqs[&b'A'], 5.0);
    /// assert_eq!(freqs[&b'C'], 5.0);
    /// assert_eq!(freqs[&b'G'], 5.0);
    /// assert_eq!(freqs[&b'T'], 5.0);
    /// assert_eq!(freqs.clone().into_values().sum::<f64>(), 20.0);
    /// ```
    pub fn get_counts(&self, alphabet: &Alphabet) -> HashMap<u8, f64> {
        let mut freqs = HashMap::new();
        for char in alphabet
            .symbols
            .iter()
            .map(|x| x as u8)
            .collect::<Vec<u8>>()
        {
            freqs.insert(
                char,
                self.sequences
                    .iter()
                    .map(|rec| rec.seq().iter().filter(|&c| c == &char).count())
                    .sum::<usize>() as f64,
            );
        }
        freqs
    }
}

/// Creates a vector of leaf encodings from the MSA, if it is provided.
/// Used for the likelihood calculation to avoid having to get the character encoding
/// from scratch every time the likelihood is optimised.
fn create_leaf_encoding(
    msa: &Option<Vec<Record>>,
    sequence_type: &SequenceType,
    gap_handling: &GapHandling,
) -> Vec<DMatrix<f64>> {
    match msa {
        None => {
            warn!("No MSA provided, leaf encoding will be empty.");
            Vec::new()
        }
        Some(msa) => {
            let mut leaf_encoding = Vec::with_capacity(msa.len());
            for seq in msa.iter() {
                leaf_encoding.push(DMatrix::from_columns(
                    seq.seq()
                        .iter()
                        .map(|&c| get_leaf_encoding(c, sequence_type, gap_handling))
                        .collect::<Vec<_>>()
                        .as_slice(),
                ));
            }
            leaf_encoding
        }
    }
}

/// Returns the character encoding for the given character.
fn get_leaf_encoding(
    char: u8,
    sequence_type: &SequenceType,
    gap_handling: &GapHandling,
) -> DVector<f64> {
    match sequence_type {
        SequenceType::DNA => match gap_handling {
            GapHandling::Ambiguous => DNA_SETS[char as usize].clone(),
            GapHandling::Proper => DNA_GAP_SETS[char as usize].clone(),
        },
        SequenceType::Protein => match gap_handling {
            GapHandling::Ambiguous => PROTEIN_SETS[char as usize].clone(),
            GapHandling::Proper => PROTEIN_GAP_SETS[char as usize].clone(),
        },
    }
}

/// Creates a PhyloInfo struct from a vector of fasta records and a tree.
/// The sequences might not be aligned, but the ids of the tree leaves and provided sequences must match.
/// In the output the sequences are sorted by the leaf ids and converted to uppercase.
///
/// # Arguments
/// * `sequences` - Vector of fasta records.
/// * `tree` - Phylogenetic tree.
/// * `gap_handling` - Gap handling option -- treat gaps as ambiguous characters or as a separate character.
///
/// # Example
/// ```
/// # use bio::io::fasta::Record;
/// # use phylo::tree::Tree;
/// # fn make_test_data() -> (Vec<Record>, Tree) {
/// #   use phylo::tree::NodeIdx::{Internal as I, Leaf as L};
/// #   let sequences = vec![
/// #       Record::with_attrs("A", None, b"aaaa"),
/// #       Record::with_attrs("B", None, b"cccc"),
/// #       Record::with_attrs("C", None, b"gg"),
/// #       Record::with_attrs("D", None, b"TTTTTTT"),
/// #   ];
/// #   let mut tree = Tree::new(&sequences).unwrap();
/// #   tree.add_parent(0, L(0), L(1), 2.0, 2.0);
/// #   tree.add_parent(1, I(0), L(2), 1.0, 2.0);
/// #   tree.add_parent(2, I(1), L(3), 1.0, 2.0);
/// #   tree.complete = true;
/// #   tree.create_postorder();
/// #   tree.create_preorder();
/// #   (sequences, tree)
/// # }
/// use phylo::phylo_info::GapHandling;
/// use phylo::phylo_info::phyloinfo_from_sequences_tree;
/// let (sequences, tree) = make_test_data();
/// let info = phyloinfo_from_sequences_tree(&sequences, tree, &GapHandling::Ambiguous).unwrap();
/// assert!(info.msa.is_none());
/// for (i, node) in info.tree.leaves.iter().enumerate() {
///     assert!(info.sequences[i].id() == node.id);
/// }
/// for rec in info.sequences.iter() {
///     assert!(!rec.seq().is_empty());
///     assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
/// }
/// ```
pub fn phyloinfo_from_sequences_tree(
    sequences: &[Record],
    tree: Tree,
    gap_handling: &GapHandling,
) -> Result<PhyloInfo> {
    let mut sequences = sequences.to_vec();
    check_sequences_not_empty(&sequences)?;
    sequences = make_sequences_uppercase(&sequences);

    validate_tree_sequence_ids(&tree, &sequences)?;
    sort_sequences_by_leaf_ids(&tree, &mut sequences);

    let msa = get_msa_if_aligned(&sequences);
    let leaf_encoding = create_leaf_encoding(&msa, &get_sequence_type(&sequences), gap_handling);
    Ok(PhyloInfo {
        sequences,
        tree,
        msa,
        leaf_encoding,
    })
}

/// Creates a PhyloInfo struct from a two given files, one containing the sequences in fasta format and
/// one containing the tree in newick format.
/// The sequences might not be aligned, but the ids of the tree leaves and provided sequences must match.
/// In the output the sequences are sorted by the leaf ids and converted to uppercase.
///
/// # Arguments
/// * `sequence_file` - File path to the sequence fasta file.
/// * `tree_file` - File path to the tree newick file.
/// * `gap_handling` - Gap handling option -- treat gaps as ambiguous characters or as a separate character.
///
/// # Example
/// ```
/// use std::path::PathBuf;
/// use phylo::phylo_info::GapHandling;
/// use phylo::phylo_info::phyloinfo_from_files;
/// let info = phyloinfo_from_files(
///     PathBuf::from("./data/sequences_DNA_small.fasta"),
///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"), &GapHandling::Ambiguous).unwrap();
/// assert!(info.msa.is_some());
/// for (i, node) in info.tree.leaves.iter().enumerate() {
///     assert!(info.sequences[i].id() == node.id);
/// }
/// for rec in info.sequences.iter() {
///     assert!(!rec.seq().is_empty());
///     assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
/// }
/// ```
pub fn phyloinfo_from_files(
    sequence_file: PathBuf,
    tree_file: PathBuf,
    gap_handling: &GapHandling,
) -> Result<PhyloInfo> {
    info!("Reading sequences from file {}", sequence_file.display());
    let mut sequences = io::read_sequences_from_file(sequence_file)?;
    info!("{} sequence(s) read successfully", sequences.len());
    check_sequences_not_empty(&sequences)?;
    sequences = make_sequences_uppercase(&sequences);

    info!("Reading trees from file {}", tree_file.display());
    let mut trees = io::read_newick_from_file(tree_file)?;
    info!("{} tree(s) read successfully", trees.len());

    check_tree_number(&trees)?;
    let tree = trees.remove(0);

    validate_tree_sequence_ids(&tree, &sequences)?;
    sort_sequences_by_leaf_ids(&tree, &mut sequences);

    let msa = get_msa_if_aligned(&sequences);
    let leaf_encoding = create_leaf_encoding(&msa, &get_sequence_type(&sequences), gap_handling);
    Ok(PhyloInfo {
        sequences,
        tree,
        msa,
        leaf_encoding,
    })
}

/// Returns a vector of records representing the MSA if all the sequences are of the same length.
/// Otherwise returns None.
///
/// # Arguments
/// * `sequences` - Vector of fasta records.
fn get_msa_if_aligned(sequences: &[Record]) -> Option<Vec<Record>> {
    let sequence_length = sequences[0].seq().len();
    if sequences
        .iter()
        .filter(|rec| rec.seq().len() != sequence_length)
        .count()
        == 0
    {
        Some(sequences.to_vec())
    } else {
        None
    }
}

/// Checks that there are sequences in the vector, bails with an error otherwise.
fn check_sequences_not_empty(sequences: &Vec<Record>) -> Result<()> {
    if sequences.is_empty() {
        bail!(DataError {
            message: String::from("No sequences provided, aborting.")
        });
    }
    Ok(())
}

/// Sorts sequences in the vector to match the order of the leaves in the tree.
fn sort_sequences_by_leaf_ids(tree: &Tree, sequences: &mut [Record]) {
    let id_index: HashMap<&str, usize> = tree
        .leaves
        .iter()
        .enumerate()
        .map(|(index, leaf)| (leaf.id.as_str(), index))
        .collect();
    sequences.sort_by_key(|record| {
        id_index
            .get(record.id())
            .cloned()
            .unwrap_or(std::usize::MAX)
    });
}

/// Checks that the ids of the tree leaves and the sequences match, bails with an error otherwise.
fn validate_tree_sequence_ids(tree: &Tree, sequences: &[Record]) -> Result<()> {
    let tip_ids: HashSet<String> = HashSet::from_iter(tree.get_leaf_ids());
    let sequence_ids: HashSet<String> =
        HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
    info!("Checking that tree tip and sequence IDs match");
    if (&tip_ids | &sequence_ids) != (&tip_ids & &sequence_ids) {
        let discrepancy: HashSet<_> = tip_ids.symmetric_difference(&sequence_ids).collect();
        bail!(DataError {
            message: format!("Mismatched IDs found: {:?}", discrepancy)
        });
    }
    Ok(())
}

/// Checks that there is at least one tree in the vector, bails with an error otherwise.
/// Prints a warning if there is more than one tree because only the first tree will be processed.
fn check_tree_number(trees: &Vec<Tree>) -> Result<()> {
    if trees.is_empty() {
        bail!(DataError {
            message: String::from("No trees in the tree file, aborting.")
        });
    }

    if trees.len() > 1 {
        warn!("More than one tree in the tree file, only the first tree will be processed.");
    }

    Ok(())
}

#[cfg(test)]
mod phylo_info_tests;
