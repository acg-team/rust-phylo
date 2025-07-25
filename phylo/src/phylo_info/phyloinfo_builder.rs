use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Ok};
use log::{info, warn};

use crate::alignment::{Aligner, Alignment, AncestralAlignment, Sequences, MASA, MSA};
use crate::alphabets::Alphabet;
use crate::asr::AncestralSequenceReconstruction;
use crate::io::{self, DataError};
use crate::parsimony::ParsimonyAligner;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
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
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new("./examples/data/sequences_DNA_small.fasta");
    /// ```
    pub fn new(sequence_file: impl AsRef<Path>) -> PhyloInfoBuilder<MSA, MASA> {
        PhyloInfoBuilder {
            sequence_file: sequence_file.as_ref().to_path_buf(),
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
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::with_attrs(
    ///     "./examples/data/sequences_DNA_small.fasta",
    ///     "./examples/data/tree_diff_branch_lengths_2.newick");
    /// ```
    pub fn with_attrs(
        sequence_file: impl AsRef<Path>,
        tree_file: impl AsRef<Path>,
    ) -> PhyloInfoBuilder<MSA, MASA> {
        PhyloInfoBuilder {
            sequence_file: sequence_file.as_ref().to_path_buf(),
            tree_file: Some(tree_file.as_ref().to_path_buf()),
            aligner: None,
            asr: None,
            alphabet: None,
        }
    }
}

impl<A: Alignment, AA: AncestralAlignment> PhyloInfoBuilder<A, AA> {
    /// Sets the tree file path for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the tree file path set.
    ///
    /// # Arguments
    /// * `path` - File path to the tree newick file.
    ///
    /// # Example
    /// ```
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::alignment::{MSA, MASA};
    /// let builder = PhyloInfoBuilder::<MSA, MASA>::new("./examples/data/sequences_DNA_small.fasta")
    ///   .tree_file(Some("./examples/data/tree_diff_branch_lengths_2.newick"));
    /// ```
    pub fn tree_file(mut self, path: Option<impl AsRef<Path>>) -> PhyloInfoBuilder<A, AA> {
        self.tree_file = path.map(|p| p.as_ref().to_path_buf());
        self
    }

    /// TODO: fix docstring
    ///
    /// # Example
    /// ```
    /// use phylo::alphabets::protein_alphabet;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::alignment::{Alignment, MSA, MASA};
    /// let info = PhyloInfoBuilder::<MSA, MASA>::new("./examples/data/sequences_DNA_small.fasta").alphabet(Some(protein_alphabet())).build().unwrap();
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
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::alignment::Alignment;
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     "./examples/data/sequences_DNA_small.fasta",
    ///     "./examples/data/tree_diff_branch_lengths_2.newick")
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(info.msa.len(), 8);
    /// assert_eq!(info.msa.seq_count(), 4);
    /// assert_eq!(info.tree.leaves().len(), 4);
    /// assert_eq!(info.tree.len(), 7);
    /// ```
    pub fn build(self) -> Result<PhyloInfo<A>> {
        let sequences = self.read_sequences()?;
        let tree = match &self.tree_file {
            Some(tree_file) => self.read_tree(tree_file)?,
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
        let mut tree = match &self.tree_file {
            Some(tree_file) => self.read_tree(tree_file)?,
            None => {
                info!("Building NJ tree from sequences");
                build_nj_tree(&sequences)?
            }
        };
        let msa = if sequences.len() == tree.n {
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
                let asr = self.asr.unwrap_or(Box::new(ParsimonyPresenceAbsence {}));
                asr.reconstruct_ancestral_seqs(&leaf_msa, &tree)
            }
        } else if sequences.len() == tree.len() {
            if sequences.aligned {
                info!("Aligned sequences including ancestral sequences.");
                AA::from_aligned_with_ancestral(sequences, &tree)
            } else {
                bail!("Building an ancestral alignment from unaligned sequences (including ancestral_sequencess) is not supported");
            }
        } else {
            bail!("The number of sequences does not match the number of leaves nor the number of nodes in the tree");
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

    // TODO: this could be a function like set_missing_tree_node_ids below, or the
    // set_missing_tree_node_ids could be a method, what do we want?
    /// Checks that there is at least one tree in the vector, bails with an error otherwise.
    /// Prints a warning if there is more than one tree because only the first tree will be processed.
    fn check_tree_number(&self, trees: &[Tree]) -> Result<()> {
        if trees.is_empty() {
            bail!(DataError {
                message: String::from("No trees in the tree file, aborting")
            });
        }
        if trees.len() > 1 {
            warn!("More than one tree in the tree file, only the first tree will be processed");
        }
        Ok(())
    }

    /// Reads trees from the provided tree file and returns the first tree
    fn read_tree(&self, tree_file: &PathBuf) -> Result<Tree> {
        info!("Reading trees from file {}", tree_file.display());
        let mut trees = io::read_newick_from_file(tree_file)?;
        info!("{} tree(s) read successfully", trees.len());
        self.check_tree_number(&trees)?;
        let tree = trees.remove(0);
        Ok(tree)
    }
}

/// Sets missing ids and bails if there are duplicates among the node ids that were already set.
pub(crate) fn set_missing_tree_node_ids(tree: &Tree) -> Result<Tree> {
    let mut tree_with_all_ids = tree.clone();
    let mut seen_user_set_ids = HashSet::new();
    let mut count = 0;
    for node_idx in tree.postorder() {
        let id = tree.node_id(node_idx);
        if id.is_empty() {
            let mut new_id = format!("I{count}");
            while !seen_user_set_ids.insert(new_id.clone()) {
                count += 1;
                new_id = format!("I{count}");
            }
            tree_with_all_ids.nodes[usize::from(node_idx)].id = new_id;
        } else if !seen_user_set_ids.insert(id.to_string()) {
            bail!("Duplicate id ({id}) found in the leaves of the tree.");
        }
    }
    Ok(tree_with_all_ids)
}

/// Checks that the ids of the tree leaves and the sequences match, bails with an error otherwise.
pub(crate) fn validate_taxa_ids(tree: &Tree, sequences: &Sequences) -> Result<()> {
    let tip_ids: HashSet<String> = HashSet::from_iter(tree.leaf_ids());
    let sequence_ids: HashSet<String> =
        HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
    info!("Checking that tree tip and sequence IDs match.");
    let mut missing_tips = sequence_ids.difference(&tip_ids).collect::<Vec<_>>();
    if !missing_tips.is_empty() {
        missing_tips.sort();
        bail!(DataError {
            message: format!("Mismatched IDs found, missing tree tip IDs: {missing_tips:?}")
        });
    }
    let mut missing_seqs = tip_ids.difference(&sequence_ids).collect::<Vec<_>>();
    if !missing_seqs.is_empty() {
        missing_seqs.sort();
        bail!(DataError {
            message: format!("Mismatched IDs found, missing sequence IDs: {missing_seqs:?}")
        });
    }
    Ok(())
}

/// Checks that the ids of the tree nodes and the sequences match, bails with an error
/// otherwise.
pub(crate) fn validate_ids_with_ancestors(tree: &Tree, sequences: &Sequences) -> Result<()> {
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
            message: format!("Mismatched IDs found, missing tree IDs: {missing_nodes:?}")
        });
    }
    let mut missing_seqs = tree_ids.difference(&sequence_ids).collect::<Vec<_>>();
    if !missing_seqs.is_empty() {
        missing_seqs.sort();
        bail!(DataError {
            message: format!("Mismatched IDs found, missing sequence IDs: {missing_seqs:?}")
        });
    }
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
pub mod private_tests {
    use std::path::Path;

    use crate::{
        alignment::Sequences,
        phylo_info::{
            phyloinfo_builder::{set_missing_tree_node_ids, PhyloInfoBuilder as PIB},
            validate_ids_with_ancestors,
        },
        record_wo_desc as record, tree,
    };

    #[test]
    fn builder_setters() {
        let fasta_path = "./examples/data/sequences_DNA_small.fasta";
        let newick_path = "./examples/data/tree_diff_branch_lengths_2.newick";

        let builder = PIB::new(fasta_path);
        assert_eq!(builder.sequence_file, Path::new(fasta_path));
        assert_eq!(builder.tree_file, None);
        let builder = builder.tree_file(Some(newick_path));
        builder.tree_file.as_ref().expect("Tree file should be set");

        assert_eq!(builder.tree_file.as_ref().unwrap(), Path::new(newick_path));
        let builder = builder.tree_file(None::<&str>);
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
    fn not_valid_ids_with_ancestors() {
        // arrange
        let tree = tree!("((A1:1.0, B1:1.0) I1:1.0,(C2:1.0,(D3:1.0, E4:1.0) I9:1.0)I10:1.0):1.0;");
        let seqs = Sequences::new(vec![
            record!("A1", b"X"),
            record!("B1", b"X"),
            record!("D3", b"X"),
            record!("E4", b"X"),
            record!("I1", b"X"),
            record!("I9", b"X"),
            record!("I10", b"X"),
            record!("", b"X"),
        ]);

        // act
        let error = validate_ids_with_ancestors(&tree, &seqs).unwrap_err();

        // assert
        assert!(error.to_string().contains("[\"C2\"]"));
    }

    #[test]
    fn valid_ids_with_ancestors() {
        // arrange
        let tree = tree!("((A1:1.0, B1:1.0) I1:1.0,(C2:1.0,(D3:1.0, E4:1.0) I9:1.0)I10:1.0):1.0;");
        let seqs = Sequences::new(vec![
            record!("A1", b"X"),
            record!("B1", b"X"),
            record!("C2", b"X"),
            record!("D3", b"X"),
            record!("E4", b"X"),
            record!("I1", b"X"),
            record!("I9", b"X"),
            record!("I10", b"X"),
            record!("", b"X"),
        ]);

        // act
        let result = validate_ids_with_ancestors(&tree, &seqs);

        // assert
        assert!(result.is_ok());
    }
}
