use std::fs::{self};
use std::iter::repeat;
use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;
use nalgebra::{dmatrix, DMatrix};
use pest::error::ErrorVariant;
use rand::Rng;

use crate::alignment::Sequences;
use crate::tree::nj_matrices::NJMat;
use crate::tree::tree_parser::{self, from_newick_string, ParsingError, Rule};
use crate::tree::{
    argmin_wo_diagonal, build_nj_tree_from_matrix, build_nj_tree_w_rng_from_matrix,
    compute_distance_matrix, percentiles, percentiles_rounded, Node, NodeIdx,
    NodeIdx::Internal as I, NodeIdx::Leaf as L, Tree,
};
use crate::{cmp_f64, Rounding};

#[cfg(test)]
fn setup_test_tree() -> Tree {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"AAA"),
    ]);
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(5, &L(0), &L(1), 1.0, 1.0);
    tree.add_parent(6, &L(3), &L(4), 1.0, 1.0);
    tree.add_parent(7, &L(2), &I(6), 1.0, 1.0);
    tree.add_parent(8, &I(5), &I(7), 1.0, 1.0);

    tree.complete = true;
    tree.compute_postorder();
    tree.compute_preorder();
    tree
}

#[test]
fn single_leaf_tree_complete() {
    let sequences = Sequences::new(vec![Record::with_attrs("A0", None, b"AAAAAA")]);
    let tree = Tree::new(&sequences).unwrap();
    assert!(tree.complete);
    assert_eq!(tree.postorder.len(), 1);
    assert_eq!(tree.preorder.len(), 1);
    assert_eq!(tree.root, L(0));
    assert_eq!(tree.nodes.len(), 1);
}

#[test]
fn try_idx_by_id() {
    let tree = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;",
    ))
    .unwrap()
    .pop()
    .unwrap();
    let nodes = [
        ("A", L(3)),
        ("B", L(4)),
        ("C", L(5)),
        ("D", L(6)),
        ("E", I(2)),
        ("F", I(1)),
        ("G", I(0)),
    ];
    for (id, idx) in nodes.iter() {
        assert!(tree.try_idx(id).is_ok());
        assert_eq!(tree.try_idx(id).unwrap(), *idx);
    }
    assert!(tree.try_idx("H").is_err());
}

#[test]
fn idx_by_id_valid() {
    let tree = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;",
    ))
    .unwrap()
    .pop()
    .unwrap();
    let nodes = [
        ("A", L(3)),
        ("B", L(4)),
        ("C", L(5)),
        ("D", L(6)),
        ("E", I(2)),
        ("F", I(1)),
        ("G", I(0)),
    ];
    for (id, idx) in nodes.iter() {
        assert_eq!(tree.idx(id), *idx);
    }
}

#[test]
#[should_panic]
fn idx_by_id_invalid() {
    let tree = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;",
    ))
    .unwrap()
    .pop()
    .unwrap();
    tree.idx("H");
}

#[test]
fn subroot_preorder() {
    let tree = setup_test_tree();
    assert_eq!(tree.preorder_subroot(&I(5)), [I(5), L(0), L(1)]);
    assert_eq!(tree.preorder_subroot(&I(6)), [I(6), L(3), L(4)]);
    assert_eq!(tree.preorder_subroot(&I(7)), [I(7), L(2), I(6), L(3), L(4)]);
    assert_eq!(
        tree.preorder_subroot(&I(8)),
        [I(8), I(5), L(0), L(1), I(7), L(2), I(6), L(3), L(4)]
    );
    assert_eq!(tree.preorder_subroot(&I(8)), tree.preorder);
    assert_eq!(tree.preorder_subroot(&tree.root), tree.preorder);
}

#[test]
fn postorder() {
    let tree = setup_test_tree();
    assert_eq!(
        tree.postorder,
        [L(0), L(1), I(5), L(2), L(3), L(4), I(6), I(7), I(8)]
    );
}

#[test]
fn tree_wo_sequences() {
    let tree = Tree::new(&Sequences::new(vec![]));
    assert!(tree.is_err());
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
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b""),
        Record::with_attrs("B1", None, b""),
        Record::with_attrs("C2", None, b""),
        Record::with_attrs("D3", None, b""),
    ]);
    let nj_tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |_| 0).unwrap();
    let nodes = vec![
        Node::new_leaf(0, Some(I(4)), 1.0, "A0".to_string()),
        Node::new_leaf(1, Some(I(4)), 3.0, "B1".to_string()),
        Node::new_leaf(2, Some(I(5)), 2.0, "C2".to_string()),
        Node::new_leaf(3, Some(I(5)), 7.0, "D3".to_string()),
        Node::new_internal(4, Some(I(6)), vec![L(0), L(1)], 1.0, "".to_string()),
        Node::new_internal(5, Some(I(6)), vec![L(3), L(2)], 1.0, "".to_string()),
        Node::new_internal(6, None, vec![I(4), I(5)], 0.0, "".to_string()),
    ];

    assert_eq!(nj_tree.root, I(6));
    assert_eq!(nj_tree.nodes, nodes);
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
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b""),
        Record::with_attrs("B1", None, b""),
        Record::with_attrs("C2", None, b""),
        Record::with_attrs("D3", None, b""),
        Record::with_attrs("E4", None, b""),
    ]);
    let nj_tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |l| 3 % l).unwrap();
    let nodes = vec![
        Node::new_leaf(0, Some(I(5)), 2.0, "A0".to_string()),
        Node::new_leaf(1, Some(I(5)), 3.0, "B1".to_string()),
        Node::new_leaf(2, Some(I(6)), 4.0, "C2".to_string()),
        Node::new_leaf(3, Some(I(7)), 2.0, "D3".to_string()),
        Node::new_leaf(4, Some(I(7)), 1.0, "E4".to_string()),
        Node::new_internal(5, Some(I(6)), vec![L(1), L(0)], 3.0, "".to_string()),
        Node::new_internal(6, Some(I(8)), vec![L(2), I(5)], 1.0, "".to_string()),
        Node::new_internal(7, Some(I(8)), vec![L(4), L(3)], 1.0, "".to_string()),
        Node::new_internal(8, None, vec![I(7), I(6)], 0.0, "".to_string()),
    ];
    assert_eq!(nj_tree.root, I(8));
    assert_eq!(nj_tree.nodes, nodes);
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
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b""),
        Record::with_attrs("B1", None, b""),
        Record::with_attrs("C2", None, b""),
        Record::with_attrs("D3", None, b""),
    ]);
    let tree = build_nj_tree_from_matrix(nj_distances, &sequences).unwrap();
    assert_eq!(tree.len(), 7);
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
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b""),
        Record::with_attrs("B", None, b""),
        Record::with_attrs("C", None, b""),
        Record::with_attrs("D", None, b""),
    ]);
    let tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |_| 0).unwrap();
    assert_eq!(tree.blen(&tree.idx("A")), 1.0);
    assert_eq!(tree.blen(&tree.idx("B")), 3.0);
    assert_eq!(tree.blen(&tree.idx("C")), 2.0);
    assert_eq!(tree.blen(&tree.idx("D")), 7.0);
    assert_eq!(tree.blen(&I(4)), 1.0);
    assert_eq!(tree.blen(&I(5)), 1.0);
    assert_eq!(tree.len(), 7);
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
    let sequences = Sequences::new(vec![
        Record::with_attrs("a", None, b""),
        Record::with_attrs("b", None, b""),
        Record::with_attrs("c", None, b""),
        Record::with_attrs("d", None, b""),
        Record::with_attrs("e", None, b""),
    ]);
    let tree = build_nj_tree_w_rng_from_matrix(nj_distances, &sequences, |l| l - 1).unwrap();
    assert_eq!(tree.blen(&tree.idx("a")), 2.0);
    assert_eq!(tree.blen(&tree.idx("b")), 3.0);
    assert_eq!(tree.blen(&tree.idx("c")), 4.0);
    assert_eq!(tree.blen(&tree.idx("d")), 1.0);
    assert_eq!(tree.blen(&tree.idx("e")), 1.0);
    assert_eq!(tree.blen(&I(5)), 3.0);
    assert_eq!(tree.blen(&I(6)), 2.0);
    assert_eq!(tree.blen(&I(7)), 1.0);
    assert_eq!(tree.len(), 9);
    assert_eq!(tree.postorder.len(), 9);
    assert!(is_unique(&tree.postorder));
    assert_eq!(tree.preorder.len(), 9);
    assert!(is_unique(&tree.preorder));
}

#[test]
fn newick_single_correct() {
    let trees = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;",
    ))
    .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let nodes = vec![
        Node::new_internal(0, None, vec![L(6), I(1)], 2.0, "G".to_string()),
        Node::new_internal(1, Some(I(0)), vec![L(5), I(2)], 1.0, "F".to_string()),
        Node::new_internal(2, Some(I(1)), vec![L(4), L(3)], 2.0, "E".to_string()),
        Node::new_leaf(3, Some(I(2)), 1.0, "A".to_string()),
        Node::new_leaf(4, Some(I(2)), 1.0, "B".to_string()),
        Node::new_leaf(5, Some(I(1)), 1.0, "C".to_string()),
        Node::new_leaf(6, Some(I(0)), 1.0, "D".to_string()),
    ];
    assert_eq!(trees[0].nodes, nodes);
    assert_eq!(trees[0].postorder.len(), 7);
    assert_eq!(trees[0].preorder.len(), 7);
}

#[test]
fn newick_ladder_first_correct() {
    let trees = from_newick_string(&String::from("((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0;")).unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let nodes = vec![
        Node::new_internal(0, None, vec![I(1), L(4)], 1.0, "F".to_string()),
        Node::new_internal(1, Some(I(0)), vec![L(3), L(2)], 2.0, "E".to_string()),
        Node::new_leaf(2, Some(I(1)), 1.0, "A".to_string()),
        Node::new_leaf(3, Some(I(1)), 1.0, "B".to_string()),
        Node::new_leaf(4, Some(I(0)), 1.0, "C".to_string()),
    ];
    assert_eq!(trees[0].nodes, nodes);
    assert_eq!(trees[0].postorder.len(), 5);
    assert_eq!(trees[0].preorder.len(), 5);
}

#[test]
fn newick_ladder_second_correct() {
    let trees = from_newick_string(&String::from("(A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;")).unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let nodes = vec![
        Node::new_internal(0, None, vec![L(1), I(2)], 1.0, "F".to_string()),
        Node::new_leaf(1, Some(I(0)), 1.0, "A".to_string()),
        Node::new_internal(2, Some(I(0)), vec![L(3), L(4)], 2.0, "E".to_string()),
        Node::new_leaf(3, Some(I(2)), 1.0, "B".to_string()),
        Node::new_leaf(4, Some(I(2)), 1.0, "C".to_string()),
    ];
    assert_eq!(trees[0].nodes, nodes);
    assert_eq!(trees[0].postorder.len(), 5);
    assert_eq!(trees[0].preorder.len(), 5);
    assert_relative_eq!(trees[0].height, trees[0].iter().map(|n| n.blen).sum());
}

#[test]
fn newick_ladder_big_correct() {
    let trees = tree_parser::from_newick_string(&String::from(
        "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;",
    ))
    .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    assert_eq!(trees[0].nodes.len(), 9);
    assert_eq!(trees[0].leaves().len(), 5);
    assert_eq!(trees[0].internals().len(), 4);
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
    assert_eq!(trees[0].nodes.len(), 31);
    assert_eq!(trees[0].leaves().len(), 16);
    assert_eq!(trees[0].internals().len(), 15);
}

#[test]
fn newick_complex_tree_2() {
    // tree from https://www.megasoftware.net/mega4/WebHelp/glossary/rh_newick_format.htm
    let newick = "(((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382),dog:25.46154);";
    let tree = tree_parser::from_newick_string(newick)
        .unwrap()
        .pop()
        .unwrap();
    assert!(tree.complete);
    assert_eq!(tree.nodes[usize::from(&tree.root)].blen, 0.0);
}

#[test]
fn newick_simple_balanced_correct() {
    let trees = tree_parser::from_newick_string(&String::from(
        "((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;",
    ))
    .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    let nodes = vec![
        Node::new_internal(0, None, vec![I(1), I(4)], 7.3, "G".to_string()),
        Node::new_internal(1, Some(I(0)), vec![L(2), L(3)], 5.1, "E".to_string()),
        Node::new_leaf(2, Some(I(1)), 1.0, "A".to_string()),
        Node::new_leaf(3, Some(I(1)), 2.0, "B".to_string()),
        Node::new_internal(4, Some(I(0)), vec![L(5), L(6)], 6.2, "F".to_string()),
        Node::new_leaf(5, Some(I(4)), 3.0, "C".to_string()),
        Node::new_leaf(6, Some(I(4)), 4.0, "D".to_string()),
    ];
    assert_eq!(trees[0].nodes, nodes);
    assert_eq!(trees[0].postorder.len(), 7);
    assert_eq!(trees[0].preorder.len(), 7);
}

#[test]
fn newick_tiny_correct() {
    let trees = tree_parser::from_newick_string(&String::from("A:1.0;")).unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, L(0));
    assert_eq!(trees[0].nodes.len(), 1);
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
    assert_eq!(trees[0].leaves().len(), 5);
    assert_eq!(trees[0].internals().len(), 4);
    assert_eq!(trees[1].root, I(0));
    assert_eq!(trees[1].leaves().len(), 4);
    assert_eq!(trees[1].internals().len(), 3);
    assert_eq!(trees[2].root, I(0));
    assert_eq!(trees[2].leaves().len(), 3);
    assert_eq!(trees[2].internals().len(), 2);
}

#[test]
fn newick_parse_parentheses_around_all() {
    let trees = tree_parser::from_newick_string(&String::from(
        "(((((A:1,B:1)F:1,C:2)G:1,D:3)H:1,E:4)I:1);",
    ));
    assert!(trees.is_ok());
}

#[test]
fn newick_parse_whitespace() {
    let trees = tree_parser::from_newick_string(&String::from(
        "     (     (((  (A:1   , B  :   1.0)  \n \n F:1,C:2.0   )G:1,D:3)H:+1.0  ,  E:4)   I:1)\n;\n   ",
    ));
    assert!(trees.is_ok());
    let tree0 = trees.unwrap().pop().unwrap();
    let tree1 = tree_parser::from_newick_string(&String::from(
        "(((((A:1,B:1)F:1,C:2)G:1,D:3)H:1,E:4)I:1);",
    ))
    .unwrap()
    .pop()
    .unwrap();
    assert_eq!(tree0.nodes, tree1.nodes);
}

#[test]
fn newick_parse_unrooted() {
    let trees = tree_parser::from_newick_string(&String::from(
        "((A:1.0,B:1.0)E:1.0,(C:1.0,D:1.0)F:1.0,G:4.0);",
    ));
    assert!(trees.is_ok());
    let tree = trees.unwrap().pop().unwrap();
    let nodes = vec![
        Node::new_internal(0, Some(I(7)), vec![L(1), L(2)], 1.0, "E".to_string()),
        Node::new_leaf(1, Some(I(0)), 1.0, "A".to_string()),
        Node::new_leaf(2, Some(I(0)), 1.0, "B".to_string()),
        Node::new_internal(3, Some(I(7)), vec![L(4), L(5)], 1.0, "F".to_string()),
        Node::new_leaf(4, Some(I(3)), 1.0, "C".to_string()),
        Node::new_leaf(5, Some(I(3)), 1.0, "D".to_string()),
        Node::new_leaf(6, Some(I(8)), 4.0, "G".to_string()),
        Node::new_internal(7, Some(I(8)), vec![I(0), I(3)], 0.0, "".to_string()),
        Node::new_internal(8, None, vec![I(7), L(6)], 0.0, "".to_string()),
    ];
    assert_eq!(tree.nodes, nodes);
    assert_eq!(tree.root, I(8));
}

#[test]
fn newick_parse_unrooted_long() {
    let mut trees = tree_parser::from_newick_string(&String::from(
        "((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382,dog:25.46154);",
    )).unwrap();
    assert_eq!(trees.len(), 1);
    let tree = trees.pop().unwrap();
    assert_eq!(tree.leaves().len(), 8);
    assert_eq!(tree.internals().len(), 7);
}

#[test]
fn newick_parse_unrooted_rooted_mix() {
    let trees = tree_parser::from_newick_string(&String::from(
        "((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382,dog:25.46154);
        ((((A:0.11,B:0.22)F:0.33,C:0.44)G:0.55,D:0.66)H:0.77,E:0.88)I:0.99;
        ((A:1.0,B:1.0)E:1.0,(C:1.0,D:1.0)F:1.0,G:4.0);
        (G:2,H:5,N:5);
        (G:2,H:5)N:5;",
    )).unwrap();
    assert_eq!(trees.len(), 5);
}

#[test]
fn newick_parse_phyml_output() {
    tree_parser::from_newick_string("((Gorilla:0.06683711,(Orangutan:0.21859880,Gibbon:0.31145586):0.06570906):0.03853171,Human:0.05356244,Chimpanzee:0.05417982);").unwrap();
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
    check_parsing_error(
        trees.unwrap_err(),
        &[Rule::tree, Rule::internal, Rule::label],
    );
    let trees = from_newick_string(&String::from("((A:1.0,B:1.0);"));
    check_parsing_error(trees.unwrap_err(), &[Rule::label, Rule::branch_length]);
    let trees = from_newick_string(&String::from("(:1.0,:2.0)E:5.1;"));
    check_parsing_error(
        trees.unwrap_err(),
        &[Rule::tree, Rule::internal, Rule::label],
    );
}

#[test]
fn check_getting_branch_lengths() {
    let tree = &from_newick_string(&String::from(
        "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;",
    ))
    .unwrap()[0];
    let mut lengths = tree.iter().map(|n| n.blen).collect::<Vec<f64>>();
    lengths.sort_by(cmp_f64());
    assert_eq!(lengths, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]);

    let tree = &from_newick_string(&String::from(
        "((((A:0.11,B:0.22)F:0.33,C:0.44)G:0.55,D:0.66)H:0.77,E:0.88)I:0.99;",
    ))
    .unwrap()[0];
    lengths = tree.iter().map(|n| n.blen).collect();
    lengths.sort_by(cmp_f64());
    assert_eq!(
        lengths,
        vec![0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]
    );

    let tree = &from_newick_string(&String::from(
        "((A:1.0,B:1.0)E:1.0,(C:1.0,D:1.0)F:1.0)G:1.0;",
    ))
    .unwrap()[0];
    lengths = tree.iter().map(|n| n.blen).collect();
    lengths.sort_by(cmp_f64());
    assert_eq!(
        lengths,
        repeat(1.0).take(lengths.len()).collect::<Vec<f64>>()
    );
}

#[test]
fn check_getting_branch_length_percentiles() {
    let perc_lengths =
        percentiles_rounded(&[3.5, 1.2, 3.7, 3.6, 1.1, 2.5, 2.4], 4, &Rounding::four());
    assert_eq!(perc_lengths, vec![1.44, 2.44, 3.1, 3.58]);
    let perc_lengths = percentiles(&repeat(1.0).take(7).collect::<Vec<f64>>(), 2);
    assert_eq!(perc_lengths, vec![1.0, 1.0]);
    let perc_lengths = percentiles(&[1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 8.0], 3);
    assert_eq!(perc_lengths, vec![3.25, 5.5, 6.75]);
}

#[test]
fn compute_distance_matrix_close() {
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b"C"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"CC"),
    ]);
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
    let sequences = Sequences::new(vec![
        Record::with_attrs("A0", None, b"AAAAAAAAAAAAAAAAAAAA"),
        Record::with_attrs("B1", None, b"AAAAAAAAAAAAAAAAAAAA"),
        Record::with_attrs("C2", None, b"AAAAAAAAAAAAAAAAAAAAAAAAA"),
        Record::with_attrs("D3", None, b"CAAAAAAAAAAAAAAAAAAA"),
    ]);
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
    assert_eq!(usize::from(&L(r1)), r1);
    let r2 = rand::thread_rng().gen_range(1..100);
    assert_eq!(usize::from(&I(r2)), r2);
}

#[test]
fn test_node_id_string() {
    let tree = tree_parser::from_newick_string(
        "((ant:17,(bat:31, cow:22)batcow:7)antbatcow:10,(elk:33,fox:12)elkfox:40)root:0;",
    )
    .unwrap()
    .pop()
    .unwrap();
    let ids = [
        "root",
        "antbatcow",
        "batcow",
        "elkfox",
        "ant",
        "bat",
        "cow",
        "elk",
        "fox",
    ];
    for node in &tree.nodes {
        assert!(ids.contains(&tree.node_id(&node.idx)));
    }
    let tree =
        tree_parser::from_newick_string("((ant:17,(bat:31, cow:22):7):10,(elk:33,fox:12):40):0;")
            .unwrap()
            .pop()
            .unwrap();

    for node in &tree.nodes {
        match node.idx {
            I(_) => {
                assert!(tree.node_id(&node.idx).is_empty());
            }
            L(_) => {
                assert!(ids.contains(&tree.node_id(&node.idx)));
            }
        }
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
fn test_node_idx_debug() {
    let r1 = rand::thread_rng().gen_range(1..100);
    assert_eq!(format!("{:?}", L(r1)), format!("Leaf({})", r1));
    let r2 = rand::thread_rng().gen_range(1..100);
    assert_eq!(format!("{:?}", I(r2)), format!("Int({})", r2));
}

#[test]
#[should_panic]
fn test_argmin_fail() {
    argmin_wo_diagonal(DMatrix::<f64>::from_vec(1, 1, vec![0.0]), |_| 0);
}

#[test]
fn test_to_newick_simple() {
    let tree = Tree {
        root: I(2),
        nodes: vec![
            Node::new_leaf(0, None, 1.0, "A".to_string()),
            Node::new_leaf(1, None, 5.5, "B".to_string()),
            Node::new_internal(2, None, vec![L(0), L(1)], 2.0, "C".to_string()),
        ],
        postorder: vec![L(0), L(1), I(2)],
        preorder: vec![I(2), L(0), L(1)],
        complete: false,
        n: 3,
        height: 8.5,
    };
    assert_eq!(tree.to_newick(), "((A:1,B:5.5)C:2);");
}

#[test]
fn test_from_newick_to_newick() {
    let newick0 = "(((((A:1,B:1)F:1,C:2)G:1,D:3)H:1,E:4)I:1);";
    let newick1 = "(((A:1.5,B:2.3)E:5.1,(C:3.9,D:4.8)F:6.2)G:7.3);";
    let newick2 = "((A:1,(B:1,C:1)E:2)F:1);";

    let trees =
        tree_parser::from_newick_string(format!("{}\n{}\n{}", newick0, newick1, newick2).as_str())
            .unwrap();
    assert_eq!(trees[0].to_newick(), newick0);
    assert_eq!(trees[1].to_newick(), newick1);
    assert_eq!(trees[2].to_newick(), newick2);
}

#[test]
fn test_to_newick_complex() {
    let newick = "(((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382):9.0,dog:25.46154):10.0;";
    let tree = tree_parser::from_newick_string(newick)
        .unwrap()
        .pop()
        .unwrap();
    assert!(tree.complete);
    assert_relative_eq!(tree.height, tree.iter().map(|n| n.blen).sum());
}

#[test]
fn check_same_trees_after_newick() {
    let newick = "(((A:1.5,B:2.3)E:5.1,(C:3.9,D:4.8)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    assert_eq!(tree.to_newick(), newick);
    let tree2 = from_newick_string(&tree.to_newick())
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(tree.nodes, tree2.nodes);
    assert_eq!(tree.root, tree2.root);
}

#[test]
fn test_parse_huge_newick() {
    let path = PathBuf::from(
        "./data/real_examples/initial_msa_env_aa_one_seq_pP_subtypeB.fas.timetree.nwk",
    );
    let newick = fs::read_to_string(path).unwrap();
    let trees = tree_parser::from_newick_string(&newick);
    assert!(trees.is_ok());
    let mut trees = trees.unwrap();
    assert_eq!(trees.len(), 1);
    let tree = trees.pop().unwrap();
    assert_eq!(tree.leaves().len(), 762);
    assert_eq!(tree.internals().len(), 761);
    assert!(tree.complete);
    assert_relative_eq!(tree.height, tree.iter().map(|n| n.blen).sum());
}

#[test]
fn test_generate_huge_newick() {
    let path = PathBuf::from(
        "./data/real_examples/initial_msa_env_aa_one_seq_pP_subtypeB.fas.timetree.nwk",
    );
    let newick = fs::read_to_string(path).unwrap();
    let trees = tree_parser::from_newick_string(&newick);
    let tree = trees.unwrap().pop().unwrap();
    let newick = tree.to_newick();
    assert!(newick.len() > 1000);
    let trees_parsed = tree_parser::from_newick_string(&newick);
    assert!(trees_parsed.is_ok());
}

#[test]
fn is_subtree() {
    let newick = "((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    assert!(tree.is_subtree(&tree.idx("A"), &tree.idx("E")));
    assert!(!tree.is_subtree(&tree.idx("E"), &tree.idx("A")));
    assert!(tree.is_subtree(&tree.idx("B"), &tree.idx("E")));
    assert!(!tree.is_subtree(&tree.idx("E"), &tree.idx("B")));
    assert!(tree.is_subtree(&tree.idx("C"), &tree.idx("F")));
    assert!(!tree.is_subtree(&tree.idx("F"), &tree.idx("C")));

    // siblings are not subtrees
    assert!(!tree.is_subtree(&tree.idx("A"), &tree.idx("B")));
    assert!(!tree.is_subtree(&tree.idx("B"), &tree.idx("A")));
    assert!(!tree.is_subtree(&tree.idx("C"), &tree.idx("D")));
    assert!(!tree.is_subtree(&tree.idx("D"), &tree.idx("C")));
    assert!(!tree.is_subtree(&tree.idx("E"), &tree.idx("F")));
    assert!(!tree.is_subtree(&tree.idx("F"), &tree.idx("E")));

    // disconnected nodes are not subtrees
    assert!(!tree.is_subtree(&tree.idx("A"), &tree.idx("H")));
    assert!(!tree.is_subtree(&tree.idx("H"), &tree.idx("A")));
    assert!(!tree.is_subtree(&tree.idx("B"), &tree.idx("H")));
    assert!(!tree.is_subtree(&tree.idx("H"), &tree.idx("D")));

    // each node is subtree of itself
    for node in tree.iter() {
        assert!(tree.is_subtree(&node.idx, &node.idx));
    }

    // root is subtree of no one
    for node in tree.iter() {
        if node.idx == tree.root {
            continue;
        }
        assert!(!tree.is_subtree(&tree.root, &node.idx));
    }
}

#[test]
fn spr_siblings() {
    let newick = "(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    assert!(tree.rooted_spr(&tree.idx("A"), &tree.idx("B")).is_err());
}

#[test]
fn spr_prune_root_or_children() {
    let newick = "(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    assert!(tree.rooted_spr(&tree.idx("G"), &tree.idx("B")).is_err());
    assert!(tree.rooted_spr(&tree.idx("E"), &tree.idx("B")).is_err());
    assert!(tree.rooted_spr(&tree.idx("F"), &tree.idx("B")).is_err());
}

#[test]
#[should_panic]
fn spr_prune_root_unchecked() {
    let newick = "(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    tree.rooted_spr_unchecked(&tree.idx("G"), &tree.idx("B"));
}

#[test]
#[should_panic]
fn spr_prune_root_child_unchecked() {
    let newick = "(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    tree.rooted_spr_unchecked(&tree.idx("F"), &tree.idx("B"));
}

#[test]
fn spr_regraft_root() {
    let newick = "(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    assert!(tree.rooted_spr(&tree.idx("A"), &tree.idx("G")).is_err());
}

#[test]
#[should_panic]
fn spr_regraft_root_unchecked() {
    let newick = "(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    tree.rooted_spr_unchecked(&tree.idx("B"), &tree.idx("G"));
}

#[test]
fn spr_regraft_subtree() {
    let newick = "((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    assert!(tree.rooted_spr(&tree.idx("E"), &tree.idx("B")).is_err());
}

#[test]
#[should_panic]
fn spr_regraft_subtree_unchecked() {
    let newick = "((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    tree.rooted_spr_unchecked(&tree.idx("E"), &tree.idx("B"));
}

#[test]
#[should_panic]
fn spr_regraft_siblings() {
    let newick = "((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    tree.rooted_spr_unchecked(&tree.idx("A"), &tree.idx("B"));
}

#[test]
fn spr_simple_valid() {
    let newick = "(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);";
    let tree = from_newick_string(newick).unwrap().pop().unwrap();
    let new_tree = tree.rooted_spr(&tree.idx("A"), &tree.idx("C")).unwrap();
    assert_eq!(new_tree.len(), tree.len());
    assert_relative_eq!(new_tree.height, tree.height);
    let prune_sib = new_tree.node(&tree.idx("B"));
    assert_eq!(prune_sib.blen, 6.1);
    assert_eq!(prune_sib.parent, Some(tree.idx("G")));
    let prune_gpar = new_tree.node(&tree.idx("G"));
    assert!([tree.idx("F"), tree.idx("B")].contains(&prune_gpar.children[0]));
    assert!([tree.idx("B"), tree.idx("F")].contains(&prune_gpar.children[1]));
    assert_eq!(prune_gpar.blen, 7.3);
    let regraft_par = new_tree.node(&tree.idx("F"));
    assert_eq!(regraft_par.blen, 6.2);
    assert!([tree.idx("E"), tree.idx("D")].contains(&regraft_par.children[0]));
    assert!([tree.idx("E"), tree.idx("D")].contains(&regraft_par.children[1]));
    let prune = new_tree.node(&tree.idx("A"));
    assert_eq!(prune.blen, 1.0);
    assert_eq!(prune.parent, Some(tree.idx("E")));
    let prune_par = new_tree.node(&tree.idx("E"));
    assert_eq!(prune_par.blen, 1.5);
    assert!([tree.idx("A"), tree.idx("C")].contains(&prune_par.children[0]));
    assert!([tree.idx("A"), tree.idx("C")].contains(&prune_par.children[1]));
    let regraft_sib = new_tree.node(&tree.idx("D"));
    assert_eq!(regraft_sib.blen, 4.0);
    assert_eq!(regraft_sib.parent, Some(tree.idx("F")));
}

#[test]
fn spr_broken() {
    let tree = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);",
    ))
    .unwrap()
    .pop()
    .unwrap();
    let new_tree = tree.rooted_spr(&tree.idx("A"), &tree.idx("C")).unwrap();
    let ng = new_tree.node(&tree.idx("G"));
    assert!([tree.idx("F"), tree.idx("B")].contains(&ng.children[0]));
    assert!([tree.idx("F"), tree.idx("B")].contains(&ng.children[1]));
    let nf = new_tree.node(&tree.idx("F"));
    assert!([tree.idx("E"), tree.idx("D")].contains(&nf.children[0]));
    assert!([tree.idx("E"), tree.idx("D")].contains(&nf.children[1]));
    assert_eq!(nf.parent, Some(tree.idx("G")));
    let ne = new_tree.node(&tree.idx("E"));
    assert!([tree.idx("A"), tree.idx("C")].contains(&ne.children[0]));
    assert!([tree.idx("A"), tree.idx("C")].contains(&ne.children[1]));
    assert_eq!(ne.parent, Some(tree.idx("F")));

    println!("{:?}", new_tree.nodes);
    assert_eq!(new_tree.len(), tree.len());
    assert_relative_eq!(new_tree.height, tree.height);
}
