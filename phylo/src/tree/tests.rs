use std::collections::HashSet;
use std::fs;
use std::path::Path;

use approx::assert_relative_eq;
use itertools::repeat_n;
use nalgebra::{dmatrix, DMatrix};
use pest::error::ErrorVariant;
use rand::Rng;

use crate::alignment::Sequences;
use crate::io::read_newick_from_file;
use crate::parsimony::Rounding;
use crate::tree::{
    argmin_wo_diagonal, build_nj_tree_from_matrix, compute_distance_matrix,
    nj_matrices::NJMat,
    percentiles, percentiles_rounded,
    tree_parser::{from_newick, ParsingError, Rule},
    Node,
    NodeIdx::{self, Internal as I, Leaf as L},
    Tree,
};
use crate::{record_wo_desc as record, tree};

#[cfg(test)]
fn setup_test_tree() -> Tree {
    let sequences = Sequences::new(vec![
        record!("A0", b"AAAAA"),
        record!("B1", b"A"),
        record!("C2", b"AA"),
        record!("D3", b"A"),
        record!("E4", b"AAA"),
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
    let sequences = Sequences::new(vec![record!("A0", b"AAAAAA")]);
    let tree = Tree::new(&sequences).unwrap();
    assert!(tree.complete);
    assert_eq!(tree.postorder.len(), 1);
    assert_eq!(tree.preorder.len(), 1);
    assert_eq!(tree.root, L(0));
    assert_eq!(tree.len(), 1);
}

#[test]
fn try_idx_by_id() {
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;");
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
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;");
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
        assert_eq!(tree.by_id(id).idx, *idx);
    }
}

#[test]
#[should_panic]
fn idx_by_id_invalid() {
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;");
    tree.by_id("H");
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
        record!("A0", b""),
        record!("B1", b""),
        record!("C2", b""),
        record!("D3", b""),
    ]);
    let nj_tree = build_nj_tree_from_matrix(nj_distances, &sequences).unwrap();
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
        record!("A0", b""),
        record!("B1", b""),
        record!("C2", b""),
        record!("D3", b""),
        record!("E4", b""),
    ]);
    let nj_tree = build_nj_tree_from_matrix(nj_distances, &sequences).unwrap();
    let nodes = vec![
        Node::new_leaf(0, Some(I(5)), 2.0, "A0".to_string()),
        Node::new_leaf(1, Some(I(5)), 3.0, "B1".to_string()),
        Node::new_leaf(2, Some(I(7)), 4.0, "C2".to_string()),
        Node::new_leaf(3, Some(I(6)), 2.0, "D3".to_string()),
        Node::new_leaf(4, Some(I(6)), 1.0, "E4".to_string()),
        Node::new_internal(5, Some(I(7)), vec![L(1), L(0)], 3.0, "".to_string()),
        Node::new_internal(6, Some(I(8)), vec![L(4), L(3)], 1.0, "".to_string()),
        Node::new_internal(7, Some(I(8)), vec![I(5), L(2)], 1.0, "".to_string()),
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
        record!("A0", b""),
        record!("B1", b""),
        record!("C2", b""),
        record!("D3", b""),
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
        record!("A", b""),
        record!("B", b""),
        record!("C", b""),
        record!("D", b""),
    ]);
    let tree = build_nj_tree_from_matrix(nj_distances, &sequences).unwrap();
    assert_eq!(tree.by_id("A").blen, 1.0);
    assert_eq!(tree.by_id("B").blen, 3.0);
    assert_eq!(tree.by_id("C").blen, 2.0);
    assert_eq!(tree.by_id("D").blen, 7.0);
    assert_eq!(tree.node(&I(4)).blen, 1.0);
    assert_eq!(tree.node(&I(5)).blen, 1.0);
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
        record!("a", b""),
        record!("b", b""),
        record!("c", b""),
        record!("d", b""),
        record!("e", b""),
    ]);
    let tree = build_nj_tree_from_matrix(nj_distances, &sequences).unwrap();
    assert_eq!(tree.by_id("a").blen, 2.0);
    assert_eq!(tree.by_id("b").blen, 3.0);
    assert_eq!(tree.by_id("c").blen, 4.0);
    assert_eq!(tree.by_id("d").blen, 2.0);
    assert_eq!(tree.by_id("e").blen, 1.0);
    assert_eq!(tree.node(&I(5)).blen, 3.0);
    assert_eq!(tree.node(&I(6)).blen, 1.0);
    assert_eq!(tree.node(&I(7)).blen, 1.0);
    assert_eq!(tree.len(), 9);
    assert_eq!(tree.postorder.len(), 9);
    assert!(is_unique(&tree.postorder));
    assert_eq!(tree.preorder.len(), 9);
    assert!(is_unique(&tree.preorder));
}

#[test]
fn newick_single_correct() {
    let trees = from_newick("(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;").unwrap();
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
    let trees = from_newick("((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0;").unwrap();
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
    let trees = from_newick("(A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;").unwrap();
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
    let trees = from_newick("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;").unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    assert_eq!(trees[0].len(), 9);
    assert_eq!(trees[0].leaves().len(), 5);
    assert_eq!(trees[0].internals().len(), 4);
    assert_eq!(trees[0].postorder.len(), 9);
    assert_eq!(trees[0].preorder.len(), 9);
}

#[test]
fn newick_complex_tree_correct() {
    // tree from file samplefraction_0.99_taxa_16_treeheight_0.8_tree1_leaves.nwk
    let trees = from_newick(
            "(((15:0.0334274,4:0.0334274):0.38581,7:0.419237):0.380763,(((6:0.0973428,14:0.0973428):0.0773821,\
            (1:0.000738004,3:0.000738004):0.173987):0.548192,(((13:0.0799156,16:0.0799156):0.0667553,(5:0.123516,\
                10:0.123516):0.0231551):0.0716431,((8:0.0571164,2:0.0571164):0.0539283,(12:0.0631742,(11:0.00312848,\
                    9:0.00312848):0.0600458):0.0478705):0.107269):0.504603):0.0770827);
            ",
        )
        .unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, I(0));
    assert_eq!(trees[0].len(), 31);
    assert_eq!(trees[0].leaves().len(), 16);
    assert_eq!(trees[0].internals().len(), 15);
}

#[test]
fn newick_complex_tree_2() {
    // tree from https://www.megasoftware.net/mega4/WebHelp/glossary/rh_newick_format.htm
    let newick =
        "(((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,
    ((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382),dog:25.46154);";
    let tree = tree!(newick);
    assert!(tree.complete);
    assert_eq!(tree.node(&tree.root).blen, 0.0);
}

#[test]
fn newick_simple_balanced_correct() {
    let trees = from_newick("((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;").unwrap();
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
    let trees = from_newick("A:1.0;").unwrap();
    assert_eq!(trees.len(), 1);
    assert_eq!(trees[0].root, L(0));
    assert_eq!(trees[0].len(), 1);
}

#[test]
fn newick_multiple_correct() {
    let trees = from_newick(
        "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;\
            ((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;\
            (A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;",
    )
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
    let trees = from_newick("(((((A:1,B:1)F:1,C:2)G:1,D:3)H:1,E:4)I:1);");
    assert!(trees.is_ok());
}

#[test]
fn newick_parse_whitespace() {
    let trees = from_newick(
        "     (     (((  (A:1   , B  :   1.0)  \n \n F:1,C:2.0   )G:1,D:3)H:+1.0  ,  E:4)   I:1)\n;\n   ",
    );
    assert!(trees.is_ok());
    let tree0 = &trees.unwrap()[0];
    let tree1 = &from_newick("(((((A:1,B:1)F:1,C:2)G:1,D:3)H:1,E:4)I:1);").unwrap()[0];
    assert_eq!(tree0.nodes, tree1.nodes);
}

#[test]
fn newick_parse_unrooted() {
    let trees = from_newick("((A:1.0,B:1.0)E:1.0,(C:1.0,D:1.0)F:1.0,G:4.0);");
    assert!(trees.is_ok());
    let tree = &trees.unwrap()[0];
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
    let trees = from_newick(
        "((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382,dog:25.46154);",
    ).unwrap();
    assert_eq!(trees.len(), 1);
    let tree = &trees[0];
    assert_eq!(tree.leaves().len(), 8);
    assert_eq!(tree.internals().len(), 7);
}

#[test]
fn newick_parse_unrooted_rooted_mix() {
    let trees = from_newick(
        "((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382,dog:25.46154);
        ((((A:0.11,B:0.22)F:0.33,C:0.44)G:0.55,D:0.66)H:0.77,E:0.88)I:0.99;
        ((A:1.0,B:1.0)E:1.0,(C:1.0,D:1.0)F:1.0,G:4.0);
        (G:2,H:5,N:5);
        (G:2,H:5)N:5;",
    ).unwrap();
    assert_eq!(trees.len(), 5);
}

#[test]
fn newick_parse_phyml_output() {
    from_newick("((Gorilla:0.06683711,(Orangutan:0.21859880,Gibbon:0.31145586):0.06570906):0.03853171,Human:0.05356244,Chimpanzee:0.05417982);").unwrap();
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
    let trees = from_newick(";");
    check_parsing_error(trees.unwrap_err(), &[Rule::newick]);
    let trees = from_newick("()()();");
    check_parsing_error(
        trees.unwrap_err(),
        &[Rule::tree, Rule::internal, Rule::label],
    );
    let trees = from_newick("((A:1.0,B:1.0);");
    check_parsing_error(trees.unwrap_err(), &[Rule::label, Rule::branch_length]);
    let trees = from_newick("(:1.0,:2.0)E:5.1;");
    check_parsing_error(
        trees.unwrap_err(),
        &[Rule::tree, Rule::internal, Rule::label],
    );
}

#[test]
fn parse_scientific_floats() {
    let tree = tree!(
        "((((A:.00001,B:1.4e-10)F:2.25e3,C:-0.546)G:1.00030000,D:+003.95)H:1.0e-10,E:4.0e0)I:-.005;"
    );
    assert_eq!(tree.by_id("A").blen, 0.00001);
    assert_eq!(tree.by_id("B").blen, 1.4e-10);
    assert_eq!(tree.by_id("C").blen, -0.546);
    assert_eq!(tree.by_id("D").blen, 3.95);
    assert_eq!(tree.by_id("E").blen, 4.0);
    assert_eq!(tree.by_id("F").blen, 2.25e3);
    assert_eq!(tree.by_id("G").blen, 1.0003);
    assert_eq!(tree.by_id("H").blen, 1.0e-10);
    assert_eq!(tree.by_id("I").blen, -0.005);
}

#[test]
fn check_getting_branch_lengths() {
    let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
    let mut lengths = tree.iter().map(|n| n.blen).collect::<Vec<f64>>();
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(lengths, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0]);

    let tree = tree!("((((A:0.11,B:0.22)F:0.33,C:0.44)G:0.55,D:0.66)H:0.77,E:0.88)I:0.99;");
    lengths = tree.iter().map(|n| n.blen).collect();
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        lengths,
        vec![0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]
    );

    let tree = tree!("((A:1.0,B:1.0)E:1.0,(C:1.0,D:1.0)F:1.0)G:1.0;");
    lengths = tree.iter().map(|n| n.blen).collect();
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(lengths, repeat_n(1.0, lengths.len()).collect::<Vec<f64>>());
}

#[test]
fn check_getting_branch_length_percentiles() {
    let perc_lengths =
        percentiles_rounded(&[3.5, 1.2, 3.7, 3.6, 1.1, 2.5, 2.4], 4, &Rounding::four());
    assert_eq!(perc_lengths, vec![1.44, 2.44, 3.1, 3.58]);
    let perc_lengths = percentiles(&repeat_n(1.0, 7).collect::<Vec<f64>>(), 2);
    assert_eq!(perc_lengths, vec![1.0, 1.0]);
    let perc_lengths = percentiles(&[1.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 8.0, 8.0], 3);
    assert_eq!(perc_lengths, vec![3.25, 5.5, 6.75]);
}

#[test]
fn compute_distance_matrix_close() {
    let sequences = Sequences::new(vec![
        record!("A0", b"C"),
        record!("B1", b"A"),
        record!("C2", b"AA"),
        record!("D3", b"A"),
        record!("E4", b"CC"),
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
        record!("A0", b"AAAAAAAAAAAAAAAAAAAA"),
        record!("B1", b"AAAAAAAAAAAAAAAAAAAA"),
        record!("C2", b"AAAAAAAAAAAAAAAAAAAAAAAAA"),
        record!("D3", b"CAAAAAAAAAAAAAAAAAAA"),
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
    let tree =
        tree!("((ant:17,(bat:31, cow:22)batcow:7)antbatcow:10,(elk:33,fox:12)elkfox:40)root:0;");
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
    let tree = tree!("((ant:17,(bat:31, cow:22):7):10,(elk:33,fox:12):40):0;");
    for node in tree.iter() {
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
    argmin_wo_diagonal(DMatrix::<f64>::from_vec(1, 1, vec![0.0]));
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
        leaf_ids: vec!["A".to_string(), "B".to_string()],
        dirty: vec![false; 3],
    };
    assert_eq!(tree.to_newick(), "((A:1,B:5.5)C:2);");
}

#[test]
fn test_from_newick_to_newick() {
    let newick0 = "(((((A:1,B:1)F:1,C:2)G:1,D:3)H:1,E:4)I:1);";
    let newick1 = "(((A:1.5,B:2.3)E:5.1,(C:3.9,D:4.8)F:6.2)G:7.3);";
    let newick2 = "((A:1,(B:1,C:1)E:2)F:1);";

    let trees = from_newick(format!("{}\n{}\n{}", newick0, newick1, newick2).as_str()).unwrap();
    assert_eq!(trees[0].to_newick(), newick0);
    assert_eq!(trees[1].to_newick(), newick1);
    assert_eq!(trees[2].to_newick(), newick2);
}

#[test]
fn test_to_newick_complex() {
    let tree = tree!("(((raccoon:19.19959,bear:6.80041):0.84600,((sea_lion:11.99700, seal:12.00300):7.52973,
    ((monkey:100.85930,cat:47.14069):20.59201, weasel:18.87953):2.09460):3.87382):9.0,dog:25.46154):10.0;");
    assert!(tree.complete);
    assert_relative_eq!(tree.height, tree.iter().map(|n| n.blen).sum());
}

#[test]
fn check_same_trees_after_newick() {
    let newick = "(((A:1.5,B:2.3)E:5.1,(C:3.9,D:4.8)F:6.2)G:7.3);";
    let tree = &from_newick(newick).unwrap()[0];
    assert_eq!(tree.to_newick(), newick);
    let tree2 = &from_newick(&tree.to_newick()).unwrap()[0];
    assert_eq!(tree.nodes, tree2.nodes);
    assert_eq!(tree.root, tree2.root);
}

#[test]
fn test_parse_huge_newick() {
    let path =
        Path::new("./data/real_examples/initial_msa_env_aa_one_seq_pP_subtypeB.fas.timetree.nwk");
    let newick = fs::read_to_string(path).unwrap();
    let trees = from_newick(&newick);
    assert!(trees.is_ok());
    let trees = trees.unwrap();
    assert_eq!(trees.len(), 1);
    let tree = &trees[0];
    assert_eq!(tree.leaves().len(), 762);
    assert_eq!(tree.internals().len(), 761);
    assert!(tree.complete);
    assert_relative_eq!(tree.height, tree.iter().map(|n| n.blen).sum());
}

#[test]
fn test_regenerate_huge_newick() {
    let path =
        Path::new("./data/real_examples/initial_msa_env_aa_one_seq_pP_subtypeB.fas.timetree.nwk");
    let newick = fs::read_to_string(path).unwrap();
    let trees = from_newick(&newick);
    let tree = &trees.unwrap()[0];
    let newick = tree.to_newick();
    assert!(newick.len() > 1000);
    let trees_parsed = from_newick(&newick);
    assert!(trees_parsed.is_ok());
}

#[test]
fn is_subtree() {
    let tree = tree!("((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);");
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
    let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
    assert!(tree.rooted_spr(&tree.idx("A"), &tree.idx("B")).is_err());
}

#[test]
fn spr_prune_root_or_children() {
    let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
    assert!(tree.rooted_spr(&tree.idx("G"), &tree.idx("B")).is_err());
    assert!(tree.rooted_spr(&tree.idx("E"), &tree.idx("B")).is_err());
    assert!(tree.rooted_spr(&tree.idx("F"), &tree.idx("B")).is_err());
}

#[test]
#[should_panic]
fn spr_prune_root_unchecked() {
    let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
    tree.rooted_spr_unchecked(&tree.idx("G"), &tree.idx("B"));
}

#[test]
#[should_panic]
fn spr_prune_root_child_unchecked() {
    let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
    tree.rooted_spr_unchecked(&tree.idx("F"), &tree.idx("B"));
}

#[test]
fn spr_regraft_root() {
    let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
    assert!(tree.rooted_spr(&tree.idx("A"), &tree.idx("G")).is_err());
}

#[test]
#[should_panic]
fn spr_regraft_root_unchecked() {
    let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
    tree.rooted_spr_unchecked(&tree.idx("B"), &tree.idx("G"));
}

#[test]
fn spr_regraft_subtree() {
    let tree = tree!("((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);");
    assert!(tree.rooted_spr(&tree.idx("E"), &tree.idx("B")).is_err());
}

#[test]
#[should_panic]
fn spr_regraft_subtree_unchecked() {
    let tree = tree!("((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);");
    tree.rooted_spr_unchecked(&tree.idx("E"), &tree.idx("B"));
}

#[test]
#[should_panic]
fn spr_regraft_siblings() {
    let tree = tree!("((((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3,H:1.0)K:1.0);");
    tree.rooted_spr_unchecked(&tree.idx("A"), &tree.idx("B"));
}

#[test]
fn spr_simple_valid() {
    let tree = tree!("(((A:1.0,B:1.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3);");
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
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
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
    assert_eq!(new_tree.len(), tree.len());
    assert_relative_eq!(new_tree.height, tree.height);
}

#[test]
fn partitions_simple() {
    let tree =
        tree!("((ant:17,(bat:31, cow:22)batcow:7)antbatcow:10,(elk:33,fox:12)elkfox:40)root:0;");
    let internal_ids: Vec<String> = ["root", "antbatcow", "batcow", "elkfox"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let correct_parts: Vec<HashSet<String>> = [
        vec!["bat", "cow"],
        vec!["ant", "bat", "cow"],
        vec!["ant", "elk", "fox"],
        vec!["elk", "fox"],
    ]
    .into_iter()
    .map(|set| set.into_iter().map(&str::to_string).collect())
    .collect();

    let partitions = tree.partitions();
    for split in partitions.iter().filter(|p| p.len() > 1 && p.len() < 4) {
        for id in internal_ids.iter() {
            assert!(!split.contains(id));
        }
        assert!(correct_parts.contains(split));
    }
}

#[test]
fn partitions() {
    let tree = tree!("((((A:1.0,B:1.0):5.1,(C:3.0,D:4.0):6.2):7.3,H:1.0):1.0);");
    let correct_parts: Vec<HashSet<String>> = [
        vec!["A", "B"],
        vec!["C", "D"],
        vec!["A", "B", "H"],
        vec!["C", "D", "H"],
    ]
    .into_iter()
    .map(|set| set.into_iter().map(&str::to_string).collect())
    .collect();

    let partitions = tree.partitions();
    assert_eq!(partitions.len(), 14);
    for split in partitions.iter().filter(|p| p.len() > 1 && p.len() < 4) {
        assert!(correct_parts.contains(split));
    }
}

#[test]
fn rf_distance_web_example() {
    // Examples from https://cs.hmc.edu/~hadas/mitcompbio/treedistance.html
    let tree1 = tree!("(0, (1, (2, (3, 4))));");
    let tree2 = tree!("(0, (1, (3, (2, 4))));");
    assert_eq!(tree1.robinson_foulds(&tree2), 2);

    let tree1 = tree!("(0, ((1, (2, 3)), (7, (6, (4, 5)))));");
    let tree2 = tree!("(0, ((2, (1, 3)), (6, (4, (5, 7)))));");
    assert_eq!(tree1.robinson_foulds(&tree2), 6);
}

#[test]
fn rf_distance_simple() {
    let tree1 = tree!("(A, (B, (C, (D, E))));");
    let tree2 = tree!("(C, (D, (E, (B, A))));");
    assert_eq!(tree1.robinson_foulds(&tree2), 2);
}

#[test]
fn rf_distance_zero_diff_taxa() {
    let tree1 = tree!("(A, (B, (C, (D, E))));");
    let tree2 = tree!("(A, (B, (C, (F, G))));");
    assert_eq!(tree1.robinson_foulds(&tree2), 0);
}

#[test]
fn rf_distance_different_sizes() {
    let tree1 = tree!("(A, (B, (C, (D, E))));");
    let tree2 = tree!("((A, (B, (C, (D, E)))),(F, (G, (H, (I, J)))));");
    assert!(tree1.robinson_foulds(&tree2) == 0);
}

#[test]
fn rf_distance_non_zero_diff_taxa() {
    let tree1 = tree!("(((a,b),c), ((e, f), g));");
    let tree2 = tree!("(((a,c),b), (g, H));");
    assert_eq!(tree1.robinson_foulds(&tree2), 2);
}

#[test]
fn rf_distance_to_itself() {
    let tree = tree!("(A, (B, (C, (D, E))));");
    assert_eq!(tree.robinson_foulds(&tree), 0);
}

#[test]
fn rf_distance_against_raxml() {
    let folder = Path::new("./data/phyml_protein_example");
    let tree_orig = &read_newick_from_file(&folder.join("example_tree.newick")).unwrap()[0];
    let tree_phyml = &read_newick_from_file(&folder.join("phyml_nogap.newick")).unwrap()[0];

    let tree = &read_newick_from_file(&folder.join("test_tree_1.newick")).unwrap()[0];
    let tree_from_nj = &read_newick_from_file(&folder.join("test_tree_2.newick")).unwrap()[0];
    assert_eq!(tree_orig.robinson_foulds(tree_phyml), 4);
    assert_eq!(tree_orig.robinson_foulds(tree), 4);
    assert_eq!(tree_orig.robinson_foulds(tree_from_nj), 4);
    assert_eq!(tree_phyml.robinson_foulds(tree), 0);
    assert_eq!(tree_phyml.robinson_foulds(tree_from_nj), 0);
    assert_eq!(tree.robinson_foulds(tree_from_nj), 0);
}
