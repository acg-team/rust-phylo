use bio::io::fasta::Record;
use hashbrown::HashSet;

use crate::alignment::{AncestralAlignment, Sequences};
use crate::phylo_info::validate_ids_with_ancestors;
use crate::tkf_model::simulate_msa::Fragmentation;
use crate::tree::tree_parser::from_newick;
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::tree::Tree;
use crate::{bail, Result, REPORT_ISSUES_URL};

/// Result of a TKF simulation (indels only, or indels + substitutions) including
/// fragmentation and the tree the simulation was performed on.
pub struct TKFSimulationResult<AA: AncestralAlignment> {
    pub masa: AA,
    pub fragmentation: Fragmentation,
    pub tree: Tree,
}

impl<AA: AncestralAlignment> TKFSimulationResult<AA> {
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
        if self.tree.n - leaves_to_remove.len() < 2 {
            bail!(
                Alignment,
                "pruning empty leaves would leave fewer than two sequences with characters which is not a meaningful tree"
            );
        }

        let aligned = self.aligned_seqs();
        let pruned_tree = prune_empty_leaves_from_tree(&self.tree, &leaves_to_remove)
            .unwrap_or_else(|e| {
                panic!(
                    "Pruning empty leaves failed: {e}. This is a bug, please report it at {REPORT_ISSUES_URL}",
                )
            });
        let original_len = self.masa.len();
        let masa: AA = rebuild_masa(&aligned, &pruned_tree).unwrap_or_else(|e| {
            panic!(
                "Rebuilding MASA after pruning empty leaves failed: {e}. \
                 This is a bug, please report it at {REPORT_ISSUES_URL}",
            )
        });
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
    /// [`Display`](`std::fmt::Display`) for MASA. Node ids are resolved through the tree,
    /// since `idx_to_id` is private to the alignment module.
    fn aligned_seqs(&self) -> Sequences {
        let both_maps = self
            .masa
            .leaf_maps()
            .iter()
            .chain(self.masa.ancestral_maps().iter());
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
        Sequences::with_alphabet_unchecked(records, self.masa.seqs().alphabet())
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
    if !matches!(leaf_idx, Leaf(_)) {
        bail!(
            Tree,
            "cannot prune node '{}', it is not a leaf",
            tree.node(leaf_idx).id
        );
    }
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

/// Collapses the root if it has only a single child, transferring its branch length to that child
/// which then becomes the new root. Loops until the root has more than one child or is a leaf.
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
/// indices, leaf ids, and traversals.
///
/// You should call [`rebuild_masa`] after this to keep the alignment in sync with the rebuilt tree.
fn rebuild_tree(pruned: &Tree) -> Result<Tree> {
    let newick = pruned.to_newick();
    let mut trees = from_newick(&newick)?;
    debug_assert_eq!(trees.len(), 1);
    Ok(trees.pop().unwrap())
}

/// Rebuilds an ancestral alignment that only contains the nodes of the new (pruned) tree,
/// keeping the aligned sequences of the remaining nodes. Columns are left untouched.
/// We first must get the aligned seqs since just by using the new_tree and the original masa
/// we cannot have access to the mappings as they are indexed through idx of the original tree.
fn rebuild_masa<AA: AncestralAlignment>(aligned: &Sequences, new_tree: &Tree) -> Result<AA> {
    let records: Vec<Record> = aligned
        .into_iter()
        .filter(|record| new_tree.try_idx(record.id()).is_ok())
        .cloned()
        .collect();
    let seqs = Sequences::with_alphabet(records, aligned.alphabet())?;
    validate_ids_with_ancestors(new_tree, &seqs)?;
    Ok(AA::from_aligned_with_ancestral_unchecked(seqs, new_tree))
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use assert_matches::assert_matches;

    use crate::alignment::{Alignment, MASA};
    use crate::tree;

    use super::*;

    /// Builds a [`TKFSimulationResult`] from aligned records (leaf and ancestral) and a
    /// tree. A dummy fragmentation is used since pruning does not touch columns.
    fn make_result(tree: &Tree, records: Vec<(&str, &[u8])>) -> TKFSimulationResult<MASA> {
        let seqs = Sequences::new(
            records
                .into_iter()
                .map(|(id, seq)| crate::record!(id, None, seq))
                .collect(),
        )
        .unwrap();
        let masa = MASA::from_aligned_with_ancestral(seqs, tree).unwrap();
        TKFSimulationResult {
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
