use super::compute_distance_matrix;
use super::nj_matrices::NJMat;
use super::tree_parser::{self, from_newick_string, ParsingError, Rule};
use super::{
    build_nj_tree_from_matrix, build_nj_tree_w_rng_from_matrix, get_percentiles, Node, NodeIdx,
    NodeIdx::Internal as I, NodeIdx::Leaf as L, Tree,
};
use crate::tree::{argmin_wo_diagonal, get_percentiles_rounded};
use crate::{cmp_f64, Rounding};
use approx::relative_eq;
use bio::io::fasta::Record;
use nalgebra::{dmatrix, DMatrix};
use pest::error::ErrorVariant;
use rand::Rng;
use std::iter::repeat;

#[cfg(test)]
fn setup_test_tree() -> Tree {
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"AAA"),
    ];
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(0, L(0), L(1), 1.0, 1.0);
    tree.add_parent(1, L(3), L(4), 1.0, 1.0);
    tree.add_parent(2, L(2), I(1), 1.0, 1.0);
    tree.add_parent(3, I(0), I(2), 1.0, 1.0);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    tree
}

#[test]
fn get_idx_by_id() {
    let tree = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;",
    ))
    .unwrap()
    .pop()
    .unwrap();
    let nodes = [
        ("A", L(0)),
        ("B", L(1)),
        ("C", L(2)),
        ("D", L(3)),
        ("E", I(2)),
        ("F", I(1)),
        ("G", I(0)),
    ];
    for (id, idx) in nodes.iter() {
        assert!(tree.get_idx_by_id(id).is_ok());
        assert_eq!(tree.get_idx_by_id(id).unwrap(), *idx);
    }
    assert!(tree.get_idx_by_id("H").is_err());
}

#[test]
fn subroot_preorder() {
    let tree = setup_test_tree();
    assert_eq!(tree.preorder_subroot(I(0)), [I(0), L(0), L(1)]);
    assert_eq!(tree.preorder_subroot(I(1)), [I(1), L(3), L(4)]);
    assert_eq!(tree.preorder_subroot(I(2)), [I(2), L(2), I(1), L(3), L(4)]);
    assert_eq!(
        tree.preorder_subroot(I(3)),
        [I(3), I(0), L(0), L(1), I(2), L(2), I(1), L(3), L(4)]
    );
    assert_eq!(tree.preorder_subroot(I(3)), tree.preorder);
}

#[test]
fn postorder() {
    let tree = setup_test_tree();
    assert_eq!(
        tree.postorder,
        [L(0), L(1), I(0), L(2), L(3), L(4), I(1), I(2), I(3)]
    );
}

#[test]
fn tree_wo_sequences() {
    let tree = Tree::new(&[]);
    assert!(tree.is_err());
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        (self.idx == other.idx)
            && (self.parent == other.parent)
            && (self.children.iter().min() == other.children.iter().min())
            && (self.children.iter().max() == other.children.iter().max())
            && relative_eq!(self.blen, other.blen)
    }
}

#[test]
fn nj_correct_web_example() {
    let nj_distances = NJMat {
        idx: (0..4).map(NodeIdx::Leaf).collect(),
        distances: dmatrix![
                0.0, 4.0, 5.0, 10.0;
                4.0, 0.0, 7.0, 12.0;
                5.0, 7.0, 0.0, 9.0;
                10.0, 12.0, 9.0, 0.0],
    };
    let sequences = vec![
        Record::with_attrs("A0", None, b""),
        Record::with_attrs("B1", None, b""),
        Record::with_attrs("C2", None, b""),
        Record::with_attrs("D3", None, b""),
    ];
    let nj_tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |_| 0).unwrap();
    let leaves = vec![
        Node::new_leaf(0, Some(I(0)), 1.0, "A0".to_string()),
        Node::new_leaf(1, Some(I(0)), 3.0, "B1".to_string()),
        Node::new_leaf(2, Some(I(1)), 2.0, "C2".to_string()),
        Node::new_leaf(3, Some(I(1)), 7.0, "D3".to_string()),
    ];
    let internals = vec![
        Node::new_internal(0, Some(I(2)), vec![L(0), L(1)], 1.0, "".to_string()),
        Node::new_internal(1, Some(I(2)), vec![L(3), L(2)], 1.0, "".to_string()),
        Node::new_internal(2, None, vec![I(0), I(1)], 0.0, "".to_string()),
    ];

    assert_eq!(nj_tree.root, I(2));
    assert_eq!(nj_tree.leaves, leaves);
    assert_eq!(nj_tree.internals, internals);
}

#[test]
fn nj_correct() {
    let nj_distances = NJMat {
        idx: (0..5).map(NodeIdx::Leaf).collect(),
        distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
    };
    let sequences = vec![
        Record::with_attrs("A0", None, b""),
        Record::with_attrs("B1", None, b""),
        Record::with_attrs("C2", None, b""),
        Record::with_attrs("D3", None, b""),
        Record::with_attrs("E4", None, b""),
    ];
    let nj_tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |l| 3 % l).unwrap();
    let leaves = vec![
        Node::new_leaf(0, Some(I(0)), 2.0, "A0".to_string()),
        Node::new_leaf(1, Some(I(0)), 3.0, "B1".to_string()),
        Node::new_leaf(2, Some(I(1)), 4.0, "C2".to_string()),
        Node::new_leaf(3, Some(I(2)), 2.0, "D3".to_string()),
        Node::new_leaf(4, Some(I(2)), 1.0, "E4".to_string()),
    ];
    let internals = vec![
        Node::new_internal(0, Some(I(1)), vec![L(1), L(0)], 3.0, "".to_string()),
        Node::new_internal(1, Some(I(3)), vec![L(2), I(0)], 1.0, "".to_string()),
        Node::new_internal(2, Some(I(3)), vec![L(4), L(3)], 1.0, "".to_string()),
        Node::new_internal(3, None, vec![I(2), I(1)], 0.0, "".to_string()),
    ];
    assert_eq!(nj_tree.root, I(3));
    assert_eq!(nj_tree.leaves, leaves);
    assert_eq!(nj_tree.internals, internals);
}

#[cfg(test)]
fn is_unique<T: std::cmp::Eq + std::hash::Hash>(vec: &[T]) -> bool {
    let set: std::collections::HashSet<_> = vec.iter().collect();
    set.len() == vec.len()
}

#[test]
fn protein_nj_correct() {
    // NJ based on example sequences from "./data/sequences_protein1.fasta"
    let nj_distances = NJMat {
        idx: (0..4).map(NodeIdx::Leaf).collect(),
        distances: dmatrix![
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.2, 0.2, 0.2, 0.0],
    };
    let sequences = vec![
        Record::with_attrs("A0", None, b""),
        Record::with_attrs("B1", None, b""),
        Record::with_attrs("C2", None, b""),
        Record::with_attrs("D3", None, b""),
    ];
    let tree = build_nj_tree_from_matrix(nj_distances, &sequences).unwrap();
    assert_eq!(tree.internals.len(), 3);
    assert_eq!(tree.postorder.len(), 7);
    assert!(is_unique(&tree.postorder));
    assert_eq!(tree.preorder.len(), 7);
    assert!(is_unique(&tree.preorder));
}

#[test]
fn nj_correct_2() {
    // NJ based on example from https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/#neighbor-joining-trees
    let nj_distances = NJMat {
        idx: (0..4).map(NodeIdx::Leaf).collect(),
        distances: dmatrix![
                0.0, 4.0, 5.0, 10.0;
                4.0, 0.0, 7.0, 12.0;
                5.0, 7.0, 0.0, 9.0;
                10.0, 12.0, 9.0, 0.0],
    };
    let sequences = vec![
        Record::with_attrs("A", None, b""),
        Record::with_attrs("B", None, b""),
        Record::with_attrs("C", None, b""),
        Record::with_attrs("D", None, b""),
    ];
    let tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |_| 0).unwrap();
    assert_eq!(branch_length(&tree, "A"), 1.0);
    assert_eq!(branch_length(&tree, "B"), 3.0);
    assert_eq!(branch_length(&tree, "C"), 2.0);
    assert_eq!(branch_length(&tree, "D"), 7.0);
    assert_eq!(tree.internals[0].blen, 1.0);
    assert_eq!(tree.internals[1].blen, 1.0);
    println!("{:?}", tree.leaves);
    assert_eq!(tree.internals.len(), 3);
    assert_eq!(tree.postorder.len(), 7);
    assert!(is_unique(&tree.postorder));
    assert_eq!(tree.preorder.len(), 7);
    assert!(is_unique(&tree.preorder));
}

#[test]
fn nj_correct_wiki_example() {
    // NJ based on example from https://en.wikipedia.org/wiki/Neighbor_joining
    let nj_distances = NJMat {
        idx: (0..5).map(NodeIdx::Leaf).collect(),
        distances: dmatrix![
            0.0, 5.0, 9.0, 9.0, 8.0;
            5.0, 0.0, 10.0, 10.0, 9.0;
            9.0, 10.0, 0.0, 8.0, 7.0;
            9.0, 10.0, 8.0, 0.0, 3.0;
            8.0, 9.0, 7.0, 3.0, 0.0],
    };
    let sequences = vec![
        Record::with_attrs("a", None, b""),
        Record::with_attrs("b", None, b""),
        Record::with_attrs("c", None, b""),
        Record::with_attrs("d", None, b""),
        Record::with_attrs("e", None, b""),
    ];
    let tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |l| l - 1).unwrap();
    assert_eq!(branch_length(&tree, "a"), 2.0);
    assert_eq!(branch_length(&tree, "b"), 3.0);
    assert_eq!(branch_length(&tree, "c"), 4.0);
    assert_eq!(branch_length(&tree, "d"), 1.0);
    assert_eq!(branch_length(&tree, "e"), 1.0);
    assert_eq!(tree.internals[0].blen, 3.0);
    assert_eq!(tree.internals[1].blen, 2.0);
    assert_eq!(tree.internals[2].blen, 1.0);
    assert_eq!(tree.internals.len(), 4);
    assert_eq!(tree.postorder.len(), 9);
    assert!(is_unique(&tree.postorder));
    assert_eq!(tree.preorder.len(), 9);
    assert!(is_unique(&tree.preorder));
}

fn branch_length(tree: &Tree, id: &str) -> f64 {
    tree.leaves[usize::from(tree.get_idx_by_id(id).unwrap())].blen
}

#[test]
fn newick_single_correct() {
    let trees = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;",
    ))
    .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let leaves = vec![
        Node::new_leaf(0, Some(I(2)), 1.0, "A".to_string()),
        Node::new_leaf(1, Some(I(2)), 1.0, "B".to_string()),
        Node::new_leaf(2, Some(I(1)), 1.0, "C".to_string()),
        Node::new_leaf(3, Some(I(0)), 1.0, "D".to_string()),
    ];
    let internals = vec![
        Node::new_internal(0, None, vec![L(3), I(1)], 2.0, "G".to_string()),
        Node::new_internal(1, Some(I(0)), vec![L(2), I(2)], 1.0, "F".to_string()),
        Node::new_internal(2, Some(I(1)), vec![L(1), L(0)], 2.0, "E".to_string()),
    ];
    assert_eq!(trees[0].leaves, leaves);
    assert_eq!(trees[0].internals, internals);
    assert_eq!(trees[0].postorder.len(), 7);
    assert_eq!(trees[0].preorder.len(), 7);
}

#[test]
fn newick_ladder_first_correct() {
    let trees = from_newick_string(&String::from("((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0;")).unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let leaves = vec![
        Node::new_leaf(0, Some(I(1)), 1.0, "A".to_string()),
        Node::new_leaf(1, Some(I(1)), 1.0, "B".to_string()),
        Node::new_leaf(2, Some(I(0)), 1.0, "C".to_string()),
    ];
    let internals = vec![
        Node::new_internal(0, None, vec![I(1), L(2)], 1.0, "F".to_string()),
        Node::new_internal(1, Some(I(0)), vec![L(1), L(0)], 2.0, "E".to_string()),
    ];
    assert_eq!(trees[0].leaves, leaves);
    assert_eq!(trees[0].internals, internals);
    assert_eq!(trees[0].postorder.len(), 5);
    assert_eq!(trees[0].preorder.len(), 5);
}

#[test]
fn newick_ladder_second_correct() {
    let trees = from_newick_string(&String::from("(A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;")).unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let leaves = vec![
        Node::new_leaf(0, Some(I(0)), 1.0, "A".to_string()),
        Node::new_leaf(1, Some(I(1)), 1.0, "B".to_string()),
        Node::new_leaf(2, Some(I(1)), 1.0, "C".to_string()),
    ];
    let internals = vec![
        Node::new_internal(0, None, vec![I(1), L(0)], 1.0, "F".to_string()),
        Node::new_internal(1, Some(I(0)), vec![L(1), L(2)], 2.0, "E".to_string()),
    ];
    assert_eq!(trees[0].leaves, leaves);
    assert_eq!(trees[0].internals, internals);
    assert_eq!(trees[0].postorder.len(), 5);
    assert_eq!(trees[0].preorder.len(), 5);
}

#[test]
fn newick_ladder_big_correct() {
    let trees = tree_parser::from_newick_string(&String::from(
        "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;",
    ))
    .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    assert_eq!(trees[0].leaves.len(), 5);
    assert_eq!(trees[0].internals.len(), 4);
    assert_eq!(trees[0].postorder.len(), 9);
    assert_eq!(trees[0].preorder.len(), 9);
}

#[test]
fn newick_complex_tree_correct() {
    // tree from file samplefraction_0.99_taxa_16_treeheight_0.8_tree1_leaves.nwk
    let trees = tree_parser::from_newick_string(&String::from(
            "(((15:0.0334274,4:0.0334274):0.38581,7:0.419237):0.380763,(((6:0.0973428,14:0.0973428):0.0773821,\
            (1:0.000738004,3:0.000738004):0.173987):0.548192,(((13:0.0799156,16:0.0799156):0.0667553,(5:0.123516,\
                10:0.123516):0.0231551):0.0716431,((8:0.0571164,2:0.0571164):0.0539283,(12:0.0631742,(11:0.00312848,\
                    9:0.00312848):0.0600458):0.0478705):0.107269):0.504603):0.0770827);
            ",
        ))
        .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    assert_eq!(trees[0].leaves.len(), 16);
    assert_eq!(trees[0].internals.len(), 15);
}

#[test]
fn newick_simple_balanced_correct() {
    let trees = tree_parser::from_newick_string(&String::from(
        "((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;",
    ))
    .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let leaves = vec![
        Node::new_leaf(0, Some(I(1)), 1.0, "A".to_string()),
        Node::new_leaf(1, Some(I(1)), 2.0, "B".to_string()),
        Node::new_leaf(2, Some(I(2)), 3.0, "C".to_string()),
        Node::new_leaf(3, Some(I(2)), 4.0, "D".to_string()),
    ];
    let internals = vec![
        Node::new_internal(0, None, vec![I(1), I(2)], 7.3, "G".to_string()),
        Node::new_internal(1, Some(I(0)), vec![L(1), L(0)], 5.1, "E".to_string()),
        Node::new_internal(2, Some(I(0)), vec![L(2), L(3)], 6.2, "F".to_string()),
    ];
    assert_eq!(trees[0].leaves, leaves);
    assert_eq!(trees[0].internals, internals);
    assert_eq!(trees[0].postorder.len(), 7);
    assert_eq!(trees[0].preorder.len(), 7);
}

#[test]
fn newick_tiny_correct() {
    let trees = tree_parser::from_newick_string(&String::from("A:1.0;")).unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, L(0));
    assert_eq!(trees[0].leaves.len(), 1);
    assert_eq!(trees[0].internals.len(), 0);
}

#[test]
fn newick_multiple_correct() {
    let trees = tree_parser::from_newick_string(&String::from(
        "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;\
            ((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;\
            (A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;",
    ))
    .unwrap();
    assert_eq!(trees.len(), 3);
    assert_eq!(trees[0].root, I(0));
    assert_eq!(trees[0].leaves.len(), 5);
    assert_eq!(trees[0].internals.len(), 4);
    assert_eq!(trees[1].root, I(0));
    assert_eq!(trees[1].leaves.len(), 4);
    assert_eq!(trees[1].internals.len(), 3);
    assert_eq!(trees[2].root, I(0));
    assert_eq!(trees[2].leaves.len(), 3);
    assert_eq!(trees[2].internals.len(), 2);
}

fn make_parsing_error(rules: &[Rule]) -> ErrorVariant<Rule> {
    ErrorVariant::ParsingError {
        positives: rules.to_vec(),
        negatives: vec![],
    }
}

fn check_parsing_error(error: anyhow::Error, expected_parsing_error: &[Rule]) {
    assert_eq!(
        error.downcast_ref::<ParsingError>().unwrap().0.variant,
        make_parsing_error(expected_parsing_error)
    );
}

#[test]
fn newick_garbage() {
    let trees = from_newick_string(&String::from(";"));
    check_parsing_error(trees.unwrap_err(), &[Rule::newick]);
    let trees = from_newick_string(&String::from("()()();"));
    check_parsing_error(trees.unwrap_err(), &[Rule::internal, Rule::label]);
    let trees = from_newick_string(&String::from("((A:1.0,B:1.0);"));
    check_parsing_error(trees.unwrap_err(), &[Rule::label, Rule::branch_length]);
    let trees = from_newick_string(&String::from("((A:1.0,B:1.0));"));
    check_parsing_error(trees.unwrap_err(), &[Rule::label, Rule::branch_length]);
    let trees = from_newick_string(&String::from("(:1.0,:2.0)E:5.1;"));
    check_parsing_error(trees.unwrap_err(), &[Rule::internal, Rule::label]);
}

#[test]
fn check_getting_branch_lengths() {
    let tree = &from_newick_string(&String::from(
        "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;",
    ))
    .unwrap()[0];
    let mut lengths = tree.get_all_branch_lengths();
    lengths.sort_by(cmp_f64());
    assert_eq!(lengths, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]);

    let tree = &from_newick_string(&String::from(
        "((((A:0.11,B:0.22)F:0.33,C:0.44)G:0.55,D:0.66)H:0.77,E:0.88)I:0.99;",
    ))
    .unwrap()[0];
    let mut lengths = tree.get_all_branch_lengths();
    lengths.sort_by(cmp_f64());
    assert_eq!(
        lengths,
        vec![0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]
    );

    let tree = &from_newick_string(&String::from(
        "((A:1.0,B:1.0)E:1.0,(C:1.0,D:1.0)F:1.0)G:1.0;",
    ))
    .unwrap()[0];
    let mut lengths = tree.get_all_branch_lengths();
    lengths.sort_by(cmp_f64());
    assert_eq!(
        lengths,
        repeat(1.0).take(lengths.len()).collect::<Vec<f64>>()
    );
}

#[test]
fn check_getting_branch_length_percentiles() {
    let perc_lengths =
        get_percentiles_rounded(&[3.5, 1.2, 3.7, 3.6, 1.1, 2.5, 2.4], 4, &Rounding::four());
    assert_eq!(perc_lengths, vec![1.44, 2.44, 3.1, 3.58]);
    let perc_lengths = get_percentiles(&repeat(1.0).take(7).collect::<Vec<f64>>(), 2);
    assert_eq!(perc_lengths, vec![1.0, 1.0]);
    let perc_lengths = get_percentiles(&[1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 8.0], 3);
    assert_eq!(perc_lengths, vec![3.25, 5.5, 6.75]);
}

#[test]
fn compute_distance_matrix_close() {
    let sequences = vec![
        Record::with_attrs("A0", None, b"C"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"CC"),
    ];
    let mat = compute_distance_matrix(&sequences);
    let true_mat = dmatrix![
        0.0, 26.728641210756745, 26.728641210756745, 26.728641210756745, 0.8239592165010822;
        26.728641210756745, 0.0, 0.8239592165010822, 0.0, 26.728641210756745;
        26.728641210756745, 0.8239592165010822, 0.0, 0.8239592165010822, 26.728641210756745;
        26.728641210756745, 0.0, 0.8239592165010822, 0.0, 26.728641210756745;
        0.8239592165010822, 26.728641210756745, 26.728641210756745, 26.728641210756745, 0.0];
    assert_eq!(mat.distances, true_mat);
}

#[test]
fn compute_distance_matrix_far() {
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAAAAAAAAAAAAAAAAA"),
        Record::with_attrs("B1", None, b"AAAAAAAAAAAAAAAAAAAA"),
        Record::with_attrs("C2", None, b"AAAAAAAAAAAAAAAAAAAAAAAAA"),
        Record::with_attrs("D3", None, b"CAAAAAAAAAAAAAAAAAAA"),
    ];
    let mat = compute_distance_matrix(&sequences);
    let true_mat = dmatrix![
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.2326161962278796, 0.2326161962278796, 0.0, 0.28924686060898847;
        0.051744653615213576, 0.051744653615213576, 0.28924686060898847, 0.0];
    assert_eq!(mat.distances, true_mat);
}

#[test]
fn test_node_idx_from_usize() {
    let r1 = rand::thread_rng().gen_range(1..100);
    assert_eq!(usize::from(NodeIdx::Leaf(r1)), r1);
    let r2 = rand::thread_rng().gen_range(1..100);
    assert_eq!(usize::from(NodeIdx::Internal(r2)), r2);
}

#[test]
fn test_get_node_id_string() {
    let tree = tree_parser::from_newick_string(
        "((ant:17,(bat:31, cow:22)batcow:7)antbatcow:10,(elk:33,fox:12)elkfox:40)root:0;",
    )
    .unwrap()
    .pop()
    .unwrap();
    let internal_ids =
        ["root", "antbatcow", "batcow", "elkfox"].map(|s| format!("{}{}", " with id ", s));
    let leaf_ids = ["ant", "bat", "cow", "elk", "fox"].map(|s| format!("{}{}", " with id ", s));
    for idx in 0..tree.internals.len() {
        assert!(internal_ids.contains(&tree.get_node_id_string(&I(idx))));
    }
    for idx in 0..tree.leaves.len() {
        assert!(leaf_ids.contains(&tree.get_node_id_string(&L(idx))));
    }
    let tree =
        tree_parser::from_newick_string("((ant:17,(bat:31, cow:22):7):10,(elk:33,fox:12):40):0;")
            .unwrap()
            .pop()
            .unwrap();
    for idx in 0..tree.internals.len() {
        assert!(tree.get_node_id_string(&I(idx)).is_empty());
    }
}

#[test]
fn test_node_idx_display() {
    let r1 = rand::thread_rng().gen_range(1..100);
    assert_eq!(format!("{}", L(r1)), format!("leaf node {}", r1));
    let r2 = rand::thread_rng().gen_range(1..100);
    assert_eq!(format!("{}", I(r2)), format!("internal node {}", r2));
}

#[test]
#[should_panic]
fn test_argmin_fail() {
    argmin_wo_diagonal(DMatrix::<f64>::from_vec(1, 1, vec![0.0]), |_| 0);
}
