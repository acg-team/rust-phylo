use crate::likelihood::{setup_dna_likelihood, setup_protein_likelihood, LikelihoodCostFunction};
use crate::phylo_info::{setup_phylogenetic_info, PhyloInfo};
use crate::tree::{NodeIdx::Leaf as L, Tree};
use approx::assert_relative_eq;
use bio::io::fasta::Record;
use std::path::PathBuf;

#[cfg(test)]
fn setup_simple_phylo_info(blen_i: f64, blen_j: f64) -> PhyloInfo {
    let sequences = vec![
        Record::with_attrs("A0", None, b"A"),
        Record::with_attrs("B1", None, b"A"),
    ];
    let mut tree = Tree::new(&sequences);
    tree.add_parent(0, L(0), L(1), blen_i, blen_j);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    PhyloInfo {
        tree,
        sequences: sequences.clone(),
        msa: Some(sequences.clone()),
    }
}

#[test]
fn dna_simple_likelihood() {
    let info = setup_simple_phylo_info(1.0, 1.0);
    let mut likelihood = setup_dna_likelihood(&info, "JC69".to_string(), &[], false).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -2.5832498829317445,
        epsilon = 1e-6
    );
    let info = setup_simple_phylo_info(1.0, 2.0);
    let mut likelihood = setup_dna_likelihood(&info, "JC69".to_string(), &[], false).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -2.719098272533848,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo {
    use crate::tree::tree_parser;
    let sequences = vec![
        Record::with_attrs("one", None, b"C"),
        Record::with_attrs("two", None, b"A"),
        Record::with_attrs("three", None, b"T"),
        Record::with_attrs("four", None, b"G"),
    ];
    let newick = "((one:2,two:2):1,(three:1,four:1):2);".to_string();
    let tree = tree_parser::from_newick_string(&newick)
        .unwrap()
        .pop()
        .unwrap();
    PhyloInfo {
        tree,
        sequences: sequences.clone(),
        msa: Some(sequences.clone()),
    }
}

#[test]
fn dna_cb_example_likelihood() {
    let info = setup_cb_example_phylo_info();
    let mut likelihood = setup_dna_likelihood(
        &info,
        "tn93".to_string(),
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -17.1035117087,
        epsilon = 1e-6
    );
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo {
    use crate::tree::tree_parser;
    let sequences = vec![
        Record::with_attrs("one", None, b"T"),
        Record::with_attrs("two", None, b"C"),
        Record::with_attrs("three", None, b"A"),
        Record::with_attrs("four", None, b"C"),
        Record::with_attrs("five", None, b"C"),
    ];
    let newick = "(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);".to_string();
    let tree = tree_parser::from_newick_string(&newick)
        .unwrap()
        .pop()
        .unwrap();
    PhyloInfo {
        tree,
        sequences: sequences.clone(),
        msa: Some(sequences.clone()),
    }
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = setup_mol_evo_example_phylo_info();
    let mut likelihood = setup_dna_likelihood(&info, "k80".to_string(), &[], true).unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -7.581408,
        epsilon = 1e-6
    );
}

#[test]
fn dna_ambig_example_likelihood() {
    let info_w_x = setup_phylogenetic_info(
        PathBuf::from("./data/ambiguous_example.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .unwrap();
    let mut likelihood_w_x = setup_dna_likelihood(
        &info_w_x,
        "tn93".to_string(),
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood_w_x.compute_log_likelihood(),
        -90.1367231323,
        epsilon = 1e-6
    );
    let info_w_n = setup_phylogenetic_info(
        PathBuf::from("./data/ambiguous_example_N.fasta"),
        PathBuf::from("./data/ambiguous_example.newick"),
    )
    .unwrap();
    let mut likelihood_w_n = setup_dna_likelihood(
        &info_w_n,
        "tn93".to_string(),
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood_w_n.compute_log_likelihood(),
        -90.1367231323,
        epsilon = 1e-6
    );
    assert_relative_eq!(
        likelihood_w_x.compute_log_likelihood(),
        likelihood_w_n.compute_log_likelihood()
    );
}

#[test]
fn dna_huelsenbeck_example_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info = setup_phylogenetic_info(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .unwrap();
    let mut likelihood = setup_dna_likelihood(
        &info,
        "gtr".to_string(),
        &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        true,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood.compute_log_likelihood(),
        -216.234734,
        epsilon = 1e-3
    );
}

#[test]
fn protein_example_likelihood() {
    let info = setup_phylogenetic_info(
        PathBuf::from("./data/phyml_protein_nogap_example.fasta"),
        PathBuf::from("./data/phyml_protein_nogap_example.newick"),
    )
    .unwrap();
    // PhyML likelihood under WAG is -4505.73681
    let mut wag_likelihood = setup_protein_likelihood(&info, "wag".to_string(), true).unwrap();
    assert_relative_eq!(
        wag_likelihood.compute_log_likelihood(),
        -4511.886841309196,
        epsilon = 1e-3
    );

    // PhyML likelihood under HIVB is -4407.98923
    let mut hivb_likelihood = setup_protein_likelihood(&info, "hivb".to_string(), true).unwrap();
    assert_relative_eq!(
        hivb_likelihood.compute_log_likelihood(),
        -4575.313436534962,
        epsilon = 1e-3
    );

    // PhyML likelihood under BLOSUM62 is -4587.71053
    let mut blos_likelihood = setup_protein_likelihood(&info, "blosum".to_string(), true).unwrap();
    assert_relative_eq!(
        blos_likelihood.compute_log_likelihood(),
        -4591.890174578042,
        epsilon = 1e-3
    );
}

#[cfg(test)]
fn setup_simple_reversibility() -> Vec<PhyloInfo> {
    use crate::tree::tree_parser;
    let mut res = Vec::<PhyloInfo>::new();
    let sequences = vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ];
    res.push(PhyloInfo {
        tree: tree_parser::from_newick_string("((A:2.0,B:2.0):1.0,C:2.0):0.0;")
            .unwrap()
            .pop()
            .unwrap(),
        sequences: sequences.clone(),
        msa: Some(sequences.clone()),
    });
    res.push(PhyloInfo {
        tree: tree_parser::from_newick_string("(A:1.0,(B:2.0,C:3.0):1.0):0.0;")
            .unwrap()
            .pop()
            .unwrap(),
        sequences: sequences.clone(),
        msa: Some(sequences.clone()),
    });
    res
}

#[test]
fn simple_likelihood_reversibility() {
    let info = setup_simple_reversibility();
    let mut likelihood1 = setup_dna_likelihood(
        &info[0],
        "tn93".to_string(),
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    let mut likelihood2 = setup_dna_likelihood(
        &info[1],
        "tn93".to_string(),
        &[0.22, 0.26, 0.33, 0.19, 0.5970915, 0.2940435, 0.00135],
        false,
    )
    .unwrap();
    assert_relative_eq!(
        likelihood1.compute_log_likelihood(),
        likelihood2.compute_log_likelihood()
    );
}

#[test]
fn huelsenbeck_example_reversibility_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let info1 = setup_phylogenetic_info(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .unwrap();
    let info2 = setup_phylogenetic_info(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example_reroot.newick"),
    )
    .unwrap();
    let mut gtr_likelihood1 = setup_dna_likelihood(
        &info1,
        "gtr".to_string(),
        &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        true,
    )
    .unwrap();
    let mut gtr_likelihood2 = setup_dna_likelihood(
        &info2,
        "gtr".to_string(),
        &[0.1, 0.3, 0.4, 0.2, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        true,
    )
    .unwrap();
    assert_relative_eq!(
        gtr_likelihood1.compute_log_likelihood(),
        gtr_likelihood2.compute_log_likelihood(),
    );
    let mut k80_likelihood1 = setup_dna_likelihood(&info1, "k80".to_string(), &[], true).unwrap();
    let mut k80_likelihood2 = setup_dna_likelihood(&info2, "k80".to_string(), &[], true).unwrap();
    assert_relative_eq!(
        k80_likelihood1.compute_log_likelihood(),
        k80_likelihood2.compute_log_likelihood(),
    );
}
