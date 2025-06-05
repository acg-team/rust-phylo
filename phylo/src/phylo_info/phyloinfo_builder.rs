use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, Ok};
use log::{info, warn};

use crate::alignment::{Aligner, Alignment, AncestralAlignment, Sequences, MASA, MSA};
use crate::alphabets::Alphabet;
use crate::asr::AncestralSequenceReconstruction;
use crate::io::{self, DataError};
use crate::parsimony::ParsimonyAligner;
use crate::parsimony_indel_points::ParsimonyIndelPoints;
use crate::phylo_info::PhyloInfo;
use crate::tree::{build_nj_tree, Tree};
use crate::Result;

pub struct PhyloInfoBuilder<A: Alignment, AA: AncestralAlignment> {
    sequence_file: PathBuf,
    tree_file: Option<PathBuf>,
    // since alignment and asr is only done once, we can use dynamic dispatch
    // but since we access the alignment on a regular basis (or do we actually? Since we instead use the encoding)
    aligner: Option<Box<dyn Aligner<A>>>,
    asr: Option<Box<dyn AncestralSequenceReconstruction<A, AA>>>,
    alphabet: Option<Alphabet>,
}

impl PhyloInfoBuilder<MSA, MASA> {
    /// Creates a new empty PhyloInfoBuilder struct with only the sequence file path set.
    /// The tree file path is set to None.
    ///
    /// # Arguments
    /// * `sequence_file` - File path to the sequence fasta file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"));
    /// ```
    pub fn new(sequence_file: PathBuf) -> PhyloInfoBuilder<MSA, MASA> {
        PhyloInfoBuilder {
            sequence_file,
            tree_file: None,
            aligner: None,
            asr: None,
            alphabet: None,
        }
    }

    /// Creates a new PhyloInfoBuilder struct with the sequence and tree file paths set.
    ///
    /// # Arguments
    /// * `sequence_file` - File path to the sequence fasta file.
    /// * `tree_file` - File path to the tree newick file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA_small.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"));
    /// ```
    pub fn with_attrs(sequence_file: PathBuf, tree_file: PathBuf) -> PhyloInfoBuilder<MSA, MASA> {
        PhyloInfoBuilder {
            sequence_file,
            tree_file: Some(tree_file),
            aligner: None,
            asr: None,
            alphabet: None,
        }
    }
}

impl<A: Alignment, AA: AncestralAlignment> PhyloInfoBuilder<A, AA> {
    /// Sets the sequence file path for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the sequence file path set.
    ///
    /// # Arguments
    /// * `path` - File path to the sequence fasta file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"))
    ///    .sequence_file(PathBuf::from("./data/sequences_DNA_small.fasta"));
    /// ```
    pub fn sequence_file(mut self, path: PathBuf) -> PhyloInfoBuilder<A, AA> {
        self.sequence_file = path;
        self
    }

    /// Sets the tree file path for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the tree file path set.
    ///
    /// # Arguments
    /// * `path` - File path to the tree newick file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"))
    ///   .tree_file(Some(PathBuf::from("./data/tree_diff_branch_lengths_2.newick")));
    /// ```
    pub fn tree_file(mut self, path: Option<PathBuf>) -> PhyloInfoBuilder<A, AA> {
        self.tree_file = path;
        self
    }

    /// Sets the tree file path for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the tree file path set.
    ///
    /// # Arguments
    /// * `path` - File path to the tree newick file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::alphabets::protein_alphabet;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::alignment::Alignment;
    /// let info = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta")).alphabet(Some(protein_alphabet())).build().unwrap();
    /// assert_eq!(info.msa.alphabet(), &protein_alphabet());
    /// ```
    pub fn alphabet(mut self, alphabet: Option<Alphabet>) -> PhyloInfoBuilder<A, AA> {
        self.alphabet = alphabet;
        self
    }

    /// Builds the PhyloInfo struct from the sequence file and the tree file (if provided).
    /// If the provided tree file has more than one tree, only the first tree will be processed.
    /// If no tree file is provided, an NJ tree is built from the sequences.
    /// Bails if no sequences are provided.
    /// Bails if the IDs of the tree leaves and the sequences do not match.
    /// Bails if the sequences are not aligned.
    /// Returns the PhyloInfo struct with the model type, tree, msa and leaf encoding set.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::alignment::Alignment;
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA_small.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"))
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(info.msa.len(), 8);
    /// assert_eq!(info.msa.seq_count(), 4);
    /// assert_eq!(info.tree.leaves().len(), 4);
    /// assert_eq!(info.tree.len(), 7);
    /// ```
    pub fn build(self) -> Result<PhyloInfo<A>> {
        let sequences = self.read_sequences()?;
        sequence_ids_are_unique(&sequences)?;
        let tree = match &self.tree_file {
            Some(tree_file) => {
                let tree = read_tree(tree_file)?;
                validate_taxa_ids(&tree, &sequences)?;
                tree
            }
            None => {
                info!("Building NJ tree from sequences");
                build_nj_tree(&sequences)?
            }
        };
        let msa = if sequences.aligned {
            info!("Sequences are aligned.");
            A::from_aligned(sequences, &tree)?
        } else {
            info!("Sequences are not aligned, aligning.");
            self.aligner
                .unwrap_or(Box::new(ParsimonyAligner::default()))
                .align(&sequences, &tree)?
        };
        Ok(PhyloInfo { tree, msa })
    }

    pub fn build_with_ancestors(self) -> Result<PhyloInfo<AA>> {
        let sequences = self.read_sequences()?;
        sequence_ids_are_unique(&sequences)?;
        let mut tree = match &self.tree_file {
            Some(tree_file) => read_tree(tree_file)?,
            None => {
                info!("Building NJ tree from sequences");
                build_nj_tree(&sequences)?
            }
        };
        let msa = if sequences.len() == tree.n {
            validate_taxa_ids(&tree, &sequences)?;
            tree = set_missing_tree_node_ids(&tree)?;
            if sequences.aligned {
                info!(
                    "Aligned sequences without ancestral sequences. Inferring ancestral sequences."
                );
                AA::from_aligned(sequences, &tree)
            } else {
                info!("Sequences are not aligned, aligning.");
                let leaf_msa = self
                    .aligner
                    .unwrap_or(Box::new(ParsimonyAligner::default()))
                    .align(&sequences, &tree)?;
                info!("Ancestral sequences are not provided, inferring them.");
                let asr = self.asr.unwrap_or(Box::new(ParsimonyIndelPoints {}));
                asr.reconstruct_ancestral_seqs(&leaf_msa, &tree)
            }
        } else if sequences.len() == tree.len() {
            if sequences.aligned {
                validate_ids_with_ancestros(&tree, &sequences)?;
                info!("Aligned sequences including ancestral sequences.");
                AA::from_aligned_with_ancestral(sequences, &tree)
            } else {
                bail!("Building an ancestral alignment from unaligned sequences (including ancestral_sequencess) is not supported");
            }
        } else {
            bail!("The number of sequences does not match the number of leaves nor the number of nodes in the tree.");
        }?;

        Ok(PhyloInfo { tree, msa })
    }

    fn read_sequences(&self) -> Result<Sequences> {
        info!(
            "Reading sequences from file {}",
            self.sequence_file.display()
        );
        let sequences = if self.alphabet.is_none() {
            info!("No alphabet provided, detecting alphabet from sequences");
            Sequences::new(io::read_sequences(&self.sequence_file)?)
        } else {
            info!(
                "Using provided {} alphabet",
                self.alphabet.as_ref().unwrap()
            );
            Sequences::with_alphabet(
                io::read_sequences(&self.sequence_file)?,
                self.alphabet.unwrap(),
            )
        };
        info!("{} sequence(s) read successfully", sequences.len());
        Ok(sequences)
    }
}

/// Sets missing ids and bails if there are duplicates among the node ids that were already set.
fn set_missing_tree_node_ids(tree: &Tree) -> Result<Tree> {
    let mut tree_with_all_ids = tree.clone();
    let mut seen_user_set_ids = HashSet::new();
    let mut count = 0;
    for node_idx in tree.postorder() {
        let id = tree.node_id(node_idx);
        if id.is_empty() {
            let mut new_id = format!("I{}", count);
            while !seen_user_set_ids.insert(new_id.clone()) {
                count += 1;
                new_id = format!("I{}", count);
            }
            tree_with_all_ids.nodes[usize::from(node_idx)].id = new_id;
        } else if !seen_user_set_ids.insert(id.to_string()) {
            bail!("Duplicate id ({}) found in the leaves of the tree.", id);
        }
    }
    Ok(tree_with_all_ids)
}
/// Reads the tree and checks if the tree node ids match the sequences ids
fn read_tree(tree_file: &PathBuf) -> Result<Tree> {
    info!("Reading trees from file {}", tree_file.display());
    let mut trees = io::read_newick_from_file(tree_file)?;
    info!("{} tree(s) read successfully", trees.len());
    check_tree_number(&trees)?;
    let tree = trees.remove(0);
    Ok(tree)
}

fn sequence_ids_are_unique(sequences: &Sequences) -> Result<()> {
    let mut seen = HashSet::new();
    for record in sequences.iter() {
        let id = record.id();
        if !seen.insert(id) {
            bail!("Duplicate record id ({}) found in the sequences.", id);
        }
    }
    Ok(())
}

/// Checks that the ids of the tree nodes and the sequences match, bails with an error
/// otherwise.
fn validate_ids_with_ancestros(tree: &Tree, sequences: &Sequences) -> Result<()> {
    let tree_ids: HashSet<String> = HashSet::from_iter(
        tree.preorder()
            .iter()
            .map(|node_idx| tree.node_id(node_idx).to_string()),
    );
    let sequence_ids: HashSet<String> =
        HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
    info!("Checking that tree and sequence IDs match.");
    let mut missing_nodes = sequence_ids.difference(&tree_ids).collect::<Vec<_>>();
    if !missing_nodes.is_empty() {
        missing_nodes.sort();
        bail!(DataError {
            message: format!(
                "Mismatched IDs found, missing tree IDs: {:?}",
                missing_nodes
            )
        });
    }
    let mut missing_seqs = tree_ids.difference(&sequence_ids).collect::<Vec<_>>();
    if !missing_seqs.is_empty() {
        missing_seqs.sort();
        bail!(DataError {
            message: format!(
                "Mismatched IDs found, missing sequence IDs: {:?}",
                missing_seqs
            )
        });
    }
    Ok(())
}

/// Checks that the ids of the tree leaves and the sequences match, bails with an error otherwise.
fn validate_taxa_ids(tree: &Tree, sequences: &Sequences) -> Result<()> {
    let tip_ids: HashSet<String> = HashSet::from_iter(tree.leaf_ids());
    let sequence_ids: HashSet<String> =
        HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
    info!("Checking that tree tip and sequence IDs match.");
    let mut missing_tips = sequence_ids.difference(&tip_ids).collect::<Vec<_>>();
    if !missing_tips.is_empty() {
        missing_tips.sort();
        bail!(DataError {
            message: format!(
                "Mismatched IDs found, missing tree tip IDs: {:?}",
                missing_tips
            )
        });
    }
    let mut missing_seqs = tip_ids.difference(&sequence_ids).collect::<Vec<_>>();
    if !missing_seqs.is_empty() {
        missing_seqs.sort();
        bail!(DataError {
            message: format!(
                "Mismatched IDs found, missing sequence IDs: {:?}",
                missing_seqs
            )
        });
    }
    Ok(())
}

/// Checks that there is at least one tree in the vector, bails with an error otherwise.
/// Prints a warning if there is more than one tree because only the first tree will be processed.
fn check_tree_number(trees: &[Tree]) -> Result<()> {
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
#[cfg_attr(coverage, coverage(off))]
pub mod private_tests {
    use std::path::PathBuf;

    use crate::{
        alignment::Sequences, phylo_info::phyloinfo_builder::set_missing_tree_node_ids,
        record_wo_desc as record, tree,
    };

    use super::{sequence_ids_are_unique, PhyloInfoBuilder as PIB};

    #[test]
    fn builder_setters() {
        let fasta1 = PathBuf::from("./data/sequences_DNA_small.fasta");
        let fasta2 = PathBuf::from("./data/sequences_DNA1.fasta");
        let newick = PathBuf::from("./data/tree_diff_branch_lengths_2.newick");

        let builder = PIB::new(fasta1.clone());
        assert_eq!(builder.sequence_file, fasta1);
        let builder = builder.sequence_file(fasta2.clone());
        assert_ne!(builder.sequence_file, fasta1);
        assert_eq!(builder.sequence_file, fasta2);

        assert_eq!(builder.tree_file, None);
        let builder = builder.tree_file(Some(newick.clone()));
        assert_eq!(builder.tree_file, Some(newick));
        let builder = builder.tree_file(None);
        assert_eq!(builder.tree_file, None);
    }

    #[test]
    fn test_set_missing_tree_node_ids() {
        // arrange
        let tree = tree!("((A1:1.0, B1:1.0) I1:1.0,(C2:1.0,(D3:1.0, E4:1.0) I9:1.0):1.0):1.0;");

        // act
        let tree = set_missing_tree_node_ids(&tree).unwrap();

        // assert
        let ids = tree
            .postorder()
            .iter()
            .map(|idx| tree.node(idx).id.clone())
            .collect::<Vec<String>>();
        assert_eq!(ids.len(), tree.len());
        assert!(!ids.contains(&String::from("")));
    }

    #[test]
    fn set_missing_tree_node_ids_finds_duplicate() {
        // arrange
        let tree = tree!("((A1:1.0, B1:1.0) I1:1.0,(C2:1.0,(D3:1.0, A1:1.0) I9:1.0):1.0):1.0;");

        // act
        let error = set_missing_tree_node_ids(&tree).unwrap_err();

        // assert
        assert!(error
            .to_string()
            .contains("Duplicate id (A1) found in the leaves of the tree."))
    }

    #[test]
    fn test_seq_ids_are_uniq() {
        // arrange
        let seqs = Sequences::new(vec![
            record!("on", b"X"),
            record!("tw", b"X"),
            record!("th", b"N"),
            record!("fo", b"N"),
        ]);

        // act
        let result = sequence_ids_are_unique(&seqs);

        // assert
        assert!(result.is_ok());
    }

    #[test]
    fn test_seq_ids_are_not_uniq() {
        // arrange
        let seqs = Sequences::new(vec![
            record!("on", b"X"),
            record!("tw", b"X"),
            record!("on", b"N"),
            record!("fo", b"N"),
        ]);

        // act
        let result = sequence_ids_are_unique(&seqs);

        // assert
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Duplicate record id (on) found in the sequences."));
    }
}
