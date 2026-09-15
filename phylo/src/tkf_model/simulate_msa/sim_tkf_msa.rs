use bio::io::fasta::Record;
use hashbrown::HashSet;
use itertools::Itertools;
use rand::{Rng, RngCore, SeedableRng};

use crate::alignment::{Alignment, AlignmentSimulation, AncestralAlignment, Sequences, MASA};
use crate::alphabets::GAP;
use crate::phylo_info::validate_ids_with_ancestors;
use crate::random::RandomGenerator;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionSimulator};
use crate::tkf_model::simulate_msa::sim_tkf_indel_msa::{
    FragmentSampler, TKFIndelMSASimulationResult, TKFIndelMSASimulator,
};
use crate::tkf_model::simulate_msa::{ExpectedRootLength, Fragmentation, RootLength};
use crate::tkf_model::TKFModel;
use crate::tree::{NodeIdx, NodeIdx::Internal, NodeIdx::Leaf, Tree};
use crate::{bail, record_wo_desc as record, Result};

/// Simulates a full TKF process: first indels then substitutions.
///
/// The simulator runs the [indel simulator](TKFIndelMSASimulator) to obtain an ancestral alignment
/// that represents homology paths of character presence, then simulates substitutions along the same
/// tree for the number of columns produced by the indel simulation and finally uses the indel MSA
/// as a mask to place gaps.
/// Note, that the MASA might contain columns where the character goes extinct, i.e., all leaf
/// sequences have a gap in that column. You may want to call
/// [`AncestralAlignment::remove_extinct_columns`](`crate::alignment::AncestralAlignment::remove_extinct_columns`) on the
/// resulting alignment or
/// [`TKFMSASimulationResult::remove_extinct_columns`](`TKFMSASimulationResult::remove_extinct_columns`) on the
/// simulation result if you also care about the fragmentation and want to remove those.
pub struct TKFMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore,
{
    indel_sim: TKFIndelMSASimulator<T, R>,
    subst_sim: SubstitutionSimulator<R>,
}

impl<T, R> TKFMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore + Clone,
{
    /// Create a new TKFMSASimulator with the given indel model, substitution model, tree, RNG and
    /// max insertion length (i.e., the max number of inserted links (=fragments) in a single event;
    /// since fragments can consist of multiple characters the number of inserted characters can be
    /// longer than this max length).
    pub fn new<Q: QMatrix>(
        indel_model: T,
        subst_model: SubstModel<Q>,
        tree: Tree,
        rng: RandomGenerator<R>,
        max_insertion_length: usize,
    ) -> Self {
        let indel_sim =
            TKFIndelMSASimulator::new(indel_model, tree.clone(), rng.clone(), max_insertion_length);
        let dummy_len = 1;
        let subst_sim = SubstitutionSimulator::new(subst_model, tree, rng, dummy_len).unwrap();
        Self {
            indel_sim,
            subst_sim,
        }
    }

    /// Sets a defined root length for the simulation. If `None`, the root length is sampled.
    pub fn root_length(&mut self, root_length: RootLength) -> &mut Self {
        self.indel_sim.root_length(root_length);
        self
    }

    /// Simulates the full TKF process (indels then substitutions) and returns both
    /// the MSA and the fragmentation (right-exclusive boundaries of fragments).
    pub fn simulate_with_fragments<AA: AncestralAlignment>(&self) -> TKFMSASimulationResult<AA> {
        // First, indel simulation
        let TKFIndelMSASimulationResult {
            masa: indel_msa,
            fragmentation,
        } = self.indel_sim.simulate_with_fragments::<AA>();

        // Second, substitution simulation
        let aln_len = indel_msa.len();
        let subst_msa: AA = self
            .subst_sim
            .simulate_ancestral_alignment_with_length(aln_len);

        // Third, mask the substitution msa with gaps from the indel msa
        // Construct a sequences vector including ancestral and leaf records
        let mut combined_records: Vec<Record> = Vec::new();
        for node in self.indel_sim.tree().preorder() {
            let id = self.indel_sim.tree().node(node).id.clone();
            // get mask_seq (from indel msa) and subst_seq (from substitution msa)
            let mask_mapping = match node {
                Internal(_) => indel_msa.ancestral_map(node),
                Leaf(_) => indel_msa.leaf_map(node),
            };
            let subst_seq = match node {
                Leaf(_) => subst_msa.seqs().record_by_id(&id).seq(),
                Internal(_) => subst_msa.ancestral_seqs().record_by_id(&id).seq(),
            };
            debug_assert!(
                mask_mapping.len() == subst_seq.len(),
                "Mask and substitution sequences must be the same length"
            );
            // apply mask: if mask is a gap, put a gap; otherwise keep the subst character
            let final_seq: Vec<u8> = mask_mapping
                .iter()
                .zip(subst_seq.iter())
                .map(
                    |(mask, subst_char)| {
                        if mask.is_some() {
                            *subst_char
                        } else {
                            GAP
                        }
                    },
                )
                .collect();

            combined_records.push(record!(&id, &final_seq));
        }

        // Lastly, construct the final ancestral MSA from the combined records
        let seqs = Sequences::new(combined_records);
        let masa = AA::from_aligned_with_ancestral(seqs, self.indel_sim.tree()).unwrap();
        TKFMSASimulationResult {
            masa,
            fragmentation,
            tree: self.indel_sim.tree().clone(),
        }
    }
}

/// Result of a full TKF simulation (indels + substitutions) including fragmentation.
pub struct TKFMSASimulationResult<AA: AncestralAlignment> {
    pub masa: AA,
    pub fragmentation: Fragmentation,
    /// The tree the simulation was performed on.
    pub tree: Tree,
}

impl<AA: AncestralAlignment> TKFMSASimulationResult<AA> {
    pub fn remove_extinct_columns(&mut self) -> Result<()> {
        let keep_col_mask = self.masa.remove_extinct_columns();
        self.fragmentation.remove_cols(&keep_col_mask)?;
        self.fragmentation
            .fragmentation_works_with_ancestral_alignment(&self.masa)
    }

    pub fn prune_empty_leaves(&mut self) -> Result<()> {
        // Collect the leaves whose aligned sequence consists only of gaps. For each of them
        // the parent node is also removed from the MASA when the tree is pruned.
        let leaves_to_remove = self.all_gap_leaves();
        if leaves_to_remove.is_empty() {
            return Ok(());
        }
        // A tree with fewer than two sequences that contain characters is not meaningful.
        if self.tree.n - leaves_to_remove.len() < 2 {
            bail!(
                Alignment,
                "pruning empty leaves would leave fewer than two sequences with characters"
            );
        }

        let aligned = self.aligned_seqs();
        let pruned_tree = prune_empty_leaves_from_tree(&self.tree, &leaves_to_remove)?;
        let original_len = self.masa.len();
        let masa: AA = rebuild_masa(&aligned, &pruned_tree)?;
        debug_assert_eq!(masa.len(), original_len);
        self.tree = pruned_tree;
        self.masa = masa;
        // Pruning only removes rows, not columns, so the fragmentation stays in sync.
        Ok(())
    }

    /// Returns the leaves whose aligned sequence consists only of gaps.
    fn all_gap_leaves(&self) -> HashSet<NodeIdx> {
        self.tree
            .preorder()
            .iter()
            .filter_map(|node_idx| match node_idx {
                Leaf(_)
                    if self
                        .masa
                        .leaf_map(node_idx)
                        .iter()
                        .all(|site| site.is_none()) =>
                {
                    Some(*node_idx)
                }
                _ => None,
            })
            .collect()
    }

    /// Returns all sequences (leaf and ancestral) in aligned form, mirroring
    /// [`Display`](`std::fmt::Display`) for [`MASA`]. Node ids are resolved through the tree,
    /// since `idx_to_id` is private to the alignment module.
    fn aligned_seqs(&self) -> Sequences {
        let both_maps = self
            .masa
            .leaf_maps()
            .iter()
            .chain(self.masa.ancestral_maps().iter());
        let both_maps =
            both_maps.sorted_by(|(a, _), (b, _)| self.tree.node_id(a).cmp(self.tree.node_id(b)));
        let records = both_maps
            .map(|(node_idx, seq_map)| {
                let id = self.tree.node_id(node_idx);
                let record = match node_idx {
                    Internal(_) => self.masa.ancestral_seqs().record_by_id(id),
                    Leaf(_) => self.masa.seqs().record_by_id(id),
                };
                let aligned = crate::aligned_seq!(seq_map, record.seq());
                crate::record!(id, record.desc(), &aligned)
            })
            .collect();
        Sequences::with_alphabet(records, self.masa.seqs().alphabet())
    }
}

/// Removes the given leaves from the tree, collapsing any single-child nodes that arise
/// and transferring their branch length to the surviving child. Also collapses the root if
/// it ends up with a single child. Mirrors ete3's
/// `TreeNode.delete(preserve_branch_length=True)` used by the `prune_empty_leaves` tool.
fn prune_empty_leaves_from_tree(tree: &Tree, leaves_to_remove: &HashSet<NodeIdx>) -> Result<Tree> {
    let mut pruned = tree.clone();
    for leaf_idx in leaves_to_remove {
        remove_leaf(&mut pruned, leaf_idx)?;
    }
    collapse_root(&mut pruned);
    rebuild_tree(&pruned)
}

/// Deletes a leaf from the tree, collapsing its parent (transferring the parent's branch
/// length to the surviving child). The root, if left with a single child, is collapsed by
/// [`collapse_root`]. Relies on the tree being binary.
fn remove_leaf(tree: &mut Tree, leaf_idx: &NodeIdx) -> Result<()> {
    let parent = match tree.parent(leaf_idx) {
        Some(parent) => parent,
        None => bail!(
            Tree,
            "cannot prune leaf '{}', it is the only node of the tree",
            tree.node(leaf_idx).id
        ),
    };
    // Disconnect the leaf from its parent
    tree.node_mut(&parent)
        .children
        .retain(|child| child != leaf_idx);

    // In a binary tree the parent of a pruned leaf is left with exactly one child;
    // collapse it and transfer its branch length to the surviving child.
    if parent != tree.root {
        debug_assert_eq!(tree.node(&parent).children.len(), 1);
        let grandparent = tree.node(&parent).parent.unwrap();
        let child = tree.node(&parent).children[0];
        let parent_blen = tree.node(&parent).blen;
        {
            let child_node = tree.node_mut(&child);
            child_node.blen += parent_blen;
            child_node.parent = Some(grandparent);
        }
        {
            let grandparent_node = tree.node_mut(&grandparent);
            grandparent_node.children.retain(|c| *c != parent);
            grandparent_node.children.push(child);
        }
    }
    Ok(())
}

/// Collapses the root if it has only a single child, transferring its branch length to
/// that child which then becomes the new root.
fn collapse_root(tree: &mut Tree) {
    loop {
        let children = tree.children(&tree.root).clone();
        match children.len() {
            1 => {
                let child = children[0];
                let root_blen = tree.node(&tree.root).blen;
                {
                    let child_node = tree.node_mut(&child);
                    child_node.blen += root_blen;
                    child_node.parent = None;
                }
                tree.root = child;
            }
            _ => break,
        }
    }
}

/// Rebuilds a clean tree from a (possibly mutated) tree, dropping any nodes that are no
/// longer connected to the root. Serialization to Newick and re-parsing re-computes all
/// indices, leaf ids, and traversals. Note that branch lengths are preserved exactly, as
/// Rust's `Display` for `f64` produces the shortest representation that round-trips.
fn rebuild_tree(pruned: &Tree) -> Result<Tree> {
    let newick = pruned.to_newick();
    let mut trees = crate::tree::tree_parser::from_newick(&newick)?;
    debug_assert_eq!(trees.len(), 1);
    Ok(trees.pop().unwrap())
}

/// Rebuilds an ancestral alignment that only contains the nodes of the new (pruned) tree,
/// keeping the aligned sequences of the remaining nodes. Columns are left untouched.
fn rebuild_masa<AA: AncestralAlignment>(aligned: &Sequences, new_tree: &Tree) -> Result<AA> {
    let records: Vec<Record> = aligned
        .into_iter()
        .filter(|record| new_tree.try_idx(record.id()).is_ok())
        .cloned()
        .collect();
    let seqs = Sequences::with_alphabet(records, aligned.alphabet());
    seqs.ids_are_unique()?;
    validate_ids_with_ancestors(new_tree, &seqs)?;
    Ok(AA::from_aligned_with_ancestral_unchecked(seqs, new_tree))
}

impl<T, R> AlignmentSimulation for TKFMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore + Clone,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> AA {
        self.simulate_with_fragments::<AA>().masa
    }

    fn simulate_alignment<A: Alignment>(&self) -> A {
        self.simulate_ancestral_alignment::<MASA>()
            .into_alignment(self.indel_sim.tree())
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use assert_matches::assert_matches;
    use hashbrown::HashSet;

    use crate::alignment::{Alignment, AncestralAlignment, MASA};
    use crate::phylo_info::PhyloInfo;
    use crate::random::DefaultGenerator;
    use crate::substitution_models::{dna_models::GTR, SubstModel};
    use crate::tkf_model::{TKF91IndelModel, TKF92IndelModel};
    use crate::tree;

    use super::*;

    /// In the [`tkf92_simulation`] test, only A <-> T and G <-> C transitions are allowed.
    /// Checks that the simulation respects the mutation constraints of the GTR model used.
    /// So each column must contain at most two unique characters, which must be a valid pair.
    #[cfg(test)]
    fn check_mutation_constraints(msa: &MASA, tree: &Tree) {
        for col_idx in 0..msa.len() {
            let mut col_chars = HashSet::new();
            for node_idx in tree.preorder() {
                let node_id = tree.node_id(node_idx);
                let seq = match node_idx {
                    Leaf(_) => msa.seqs().record_by_id(node_id).seq(),
                    Internal(_) => msa.ancestral_seqs().record_by_id(node_id).seq(),
                };
                let map = match node_idx {
                    Internal(_) => msa.ancestral_map(node_idx),
                    Leaf(_) => msa.leaf_map(node_idx),
                };

                if let Some(pos) = map[col_idx] {
                    col_chars.insert(seq[pos]);
                }
            }
            assert!(
                col_chars.len() <= 2,
                "Column {} has too many unique characters: {:?}",
                col_idx,
                col_chars
            );
            // Verify specific pairings if multiple characters exist
            if col_chars.len() == 2 {
                let chars: Vec<u8> = col_chars.into_iter().collect();
                let c1 = chars[0];
                let c2 = chars[1];
                let valid_pair = matches!(
                    (c1, c2),
                    (b'A', b'T') | (b'T', b'A') | (b'C', b'G') | (b'G', b'C')
                );
                assert!(
                    valid_pair,
                    "Invalid mutation pair in column {}: {} and {}",
                    col_idx, c1 as char, c2 as char
                );
            }
        }
    }

    #[test]
    fn tkf92_simulation() {
        let tree = tree!(
            "((((A:1.0,B:1.0)I1:0.5,C:1.5)I2:0.5,D:2.0)I3:0.5,((E:1.0,F:1.0)I4:0.5,G:1.5)I5:0.5)R;"
        );

        let freqs = [0.25, 0.25, 0.25, 0.25];
        let params = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let subst_model = SubstModel::<GTR>::new(&freqs, &params);

        let lambda = 0.19;
        let mu = 0.2;
        let r = 0.8;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);

        let max_insertion_length = 50;
        let simulator = TKFMSASimulator::new(
            tkf_model,
            subst_model,
            tree.clone(),
            DefaultGenerator::new(123),
            max_insertion_length,
        );

        let msa = simulator.simulate_ancestral_alignment::<MASA>();

        assert_eq!(msa.seq_count() + msa.ancestral_seqs().len(), 13);
        assert_eq!(msa.seq_count(), 7); // A, B, C, D, E, F, G
        assert!(msa.len() > 1);

        check_mutation_constraints(&msa, &tree);

        let phylo = PhyloInfo {
            msa: msa.clone(),
            tree,
        };
        assert!(
            phylo.check_dollos_constraint().is_ok(),
            "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
        );
    }

    #[test]
    fn tkf91_simulation_fixed_root_length() {
        let tree = tree!("(A:1.0,B:1.0)R;");
        let subst_model = SubstModel::<GTR>::new(&[0.25; 4], &[1.0; 6]);
        let tkf_model = TKF91IndelModel::new(0.1, 0.2);

        let mut simulator = TKFMSASimulator::new(
            tkf_model,
            subst_model,
            tree.clone(),
            DefaultGenerator::new(123),
            50,
        );
        simulator.root_length(RootLength::Defined(100));

        let msa = simulator.simulate_ancestral_alignment::<MASA>();
        let root_map = msa.ancestral_map(&tree.root);
        assert_eq!(root_map.iter().filter(|s| s.is_some()).count(), 100);
    }

    #[test]
    fn tkf92_simulation_fixed_root_length() {
        let tree = tree!("(A:1.0,B:1.0)R;");
        let subst_model = SubstModel::<GTR>::new(&[0.25; 4], &[1.0; 6]);
        let tkf_model = TKF92IndelModel::new(0.1, 0.2, 0.8);

        let mut simulator = TKFMSASimulator::new(
            tkf_model,
            subst_model,
            tree.clone(),
            DefaultGenerator::new(123),
            50,
        );
        simulator.root_length(RootLength::Defined(100));

        let msa = simulator.simulate_ancestral_alignment::<MASA>();
        let root_map = msa.ancestral_map(&tree.root);
        assert_eq!(root_map.iter().filter(|s| s.is_some()).count(), 100);
    }

    /// Builds a [`TKFMSASimulationResult`] from aligned records (leaf and ancestral) and a
    /// tree. A dummy fragmentation is used since pruning does not touch columns.
    fn make_result(tree: &Tree, records: Vec<(&str, &[u8])>) -> TKFMSASimulationResult<MASA> {
        let seqs = Sequences::new(
            records
                .into_iter()
                .map(|(id, seq)| crate::record!(id, None, seq))
                .collect(),
        );
        let masa = MASA::from_aligned_with_ancestral(seqs, tree).unwrap();
        TKFMSASimulationResult {
            masa,
            fragmentation: Fragmentation::new(vec![10]).unwrap(),
            tree: tree.clone(),
        }
    }

    #[test]
    fn prune_empty_leaves_leaf_only() {
        // C4 consists only of gaps -> prune C4 and its parent R5, the root collapses to I3.
        let tree = tree!("(((A1:1.0,B2:2.0)I3:3.0,C4:4.0)R5:5.0);");
        let mut result = make_result(
            &tree,
            vec![
                ("A1", b"--GTGGA---"),
                ("B2", b"-------NNA"),
                ("I3", b"--T-------"),
                ("C4", b"----------"),
                ("R5", b"--A-------"),
            ],
        );
        result.prune_empty_leaves().unwrap();

        assert_eq!(result.masa.seq_count(), 2); // A1, B2
        assert_eq!(result.masa.ancestral_seqs().len(), 1); // I3
        assert_eq!(result.masa.len(), 8);
        assert_eq!(result.masa.leaf_map(&result.tree.by_id("A1").idx).len(), 8);

        assert_eq!(result.tree.leaves().len(), 2);
        assert_eq!(result.tree.node_id(&result.tree.root), "I3");
        assert_eq!(result.tree.node(&result.tree.root).blen, 8.0);
        assert_eq!(result.tree.by_id("A1").blen, 1.0);
        assert_eq!(result.tree.by_id("B2").blen, 2.0);
        assert!(result.tree.try_idx("C4").is_err());
        assert!(result.tree.try_idx("R5").is_err());
    }

    #[test]
    fn prune_empty_leaves_parent_removed() {
        // A1 consists only of gaps -> prune A1 and its parent I3; B2 takes over I3's branch.
        let tree = tree!("(((A1:1.0,B2:2.0)I3:3.0,C4:4.0)R5:5.0);");
        let mut result = make_result(
            &tree,
            vec![
                ("A1", b"----------"),
                ("B2", b"-------NNA"),
                ("I3", b"--T-------"),
                ("C4", b"------A---"),
                ("R5", b"--A-------"),
            ],
        );
        result.prune_empty_leaves().unwrap();

        assert_eq!(result.masa.seq_count(), 2); // B2, C4
        assert_eq!(result.masa.ancestral_seqs().len(), 1); // R5
        assert_eq!(result.tree.leaves().len(), 2);
        assert_eq!(result.tree.node_id(&result.tree.root), "R5");
        assert_eq!(result.tree.by_id("R5").blen, 5.0);
        assert_eq!(result.tree.by_id("B2").blen, 5.0);
        assert_eq!(result.tree.by_id("C4").blen, 4.0);
        assert!(result.tree.try_idx("A1").is_err());
        assert!(result.tree.try_idx("I3").is_err());
    }

    #[test]
    fn prune_empty_leaves_single_leaf_bails() {
        // A1 and B2 both consist only of gaps -> pruning would leave a single sequence (C4)
        // with characters, which is not a meaningful tree.
        let tree = tree!("(((A1:1.0,B2:2.0)I3:3.0,C4:4.0)R5:5.0);");
        let mut result = make_result(
            &tree,
            vec![
                ("A1", b"----------"),
                ("B2", b"----------"),
                ("I3", b"--T-------"),
                ("C4", b"------A---"),
                ("R5", b"--A-------"),
            ],
        );
        assert_matches!(
            result.prune_empty_leaves(),
            Err(crate::Error::Alignment(msg)) if msg.contains("fewer than two sequences")
        );
    }

    #[test]
    fn prune_empty_leaves_all_gap_bails() {
        // All sequences consist only of gaps -> pruning would leave no sequences with
        // characters at all.
        let tree = tree!("(((A1:1.0,B2:2.0)I3:3.0,C4:4.0)R5:5.0);");
        let mut result = make_result(
            &tree,
            vec![
                ("A1", b"----------"),
                ("B2", b"----------"),
                ("I3", b"----------"),
                ("C4", b"----------"),
                ("R5", b"----------"),
            ],
        );
        assert_matches!(
            result.prune_empty_leaves(),
            Err(crate::Error::Alignment(msg)) if msg.contains("fewer than two sequences")
        );
    }

    #[test]
    fn prune_empty_leaves_nothing_to_prune() {
        // No gap-only leaves -> the result is unchanged.
        let tree = tree!("(((A1:1.0,B2:2.0)I3:3.0,C4:4.0)R5:5.0);");
        let mut result = make_result(
            &tree,
            vec![
                ("A1", b"--GTGGA---"),
                ("B2", b"-------NNA"),
                ("I3", b"--T-------"),
                ("C4", b"------A---"),
                ("R5", b"--A-------"),
            ],
        );
        result.prune_empty_leaves().unwrap();

        assert_eq!(result.tree.to_newick(), tree.to_newick());
        assert_eq!(result.masa.seq_count(), 3);
        assert_eq!(result.masa.ancestral_seqs().len(), 2);
    }

    #[test]
    fn prune_empty_leaves_comb() {
        // G and N1 consist only of gaps -> removing G collapses I1 (N1 takes over I1's
        // branch), removing N1 collapses I2 (N2 takes over I2's branch).
        let tree = tree!("((((G:1.0,N1:1.0)I1:1.0,N2:1.0)I2:1.0,N3:1.0)I3:1.0,N4:1.0)R:1.0;");
        let mut result = make_result(
            &tree,
            vec![
                ("G", b"----"),
                ("N1", b"----"),
                ("I1", b"A---"),
                ("N2", b"A---"),
                ("I2", b"A---"),
                ("N3", b"-A--"),
                ("I3", b"A---"),
                ("N4", b"--A-"),
                ("R", b"---A"),
            ],
        );
        result.prune_empty_leaves().unwrap();

        assert_eq!(result.masa.seq_count(), 3); // N2, N3, N4
        assert_eq!(result.tree.leaves().len(), 3);
        assert_eq!(result.tree.by_id("N2").blen, 2.0);
        assert_eq!(result.tree.by_id("N3").blen, 1.0);
        assert_eq!(result.tree.by_id("N4").blen, 1.0);
        assert!(result.tree.try_idx("G").is_err());
        assert!(result.tree.try_idx("N1").is_err());
        assert!(result.tree.try_idx("I1").is_err());
        assert!(result.tree.try_idx("I2").is_err());
    }
}
