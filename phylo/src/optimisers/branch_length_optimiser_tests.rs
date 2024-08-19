use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::DNAModelType;
use crate::optimisers::branch_length_optimiser::BranchOptimiser;
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::{PIPCost, PIPDNAModel};
use crate::substitution_models::dna_models::DNASubstModel;
use crate::substitution_models::{SubstitutionLikelihoodCost, SubstitutionModel};
use crate::tree::tree_parser::from_newick_string;
use crate::tree::NodeIdx::Internal as Int;

#[test]
fn branch_optimiser_likelihood_increase_pip() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = PIPDNAModel::new(
        DNAModelType::GTR,
        &[
            14.142_1, 0.1414, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let cost = PIPCost { model: &model };
    let o = BranchOptimiser::new(&cost, &info).run().unwrap();
    assert!(o.final_logl > o.initial_logl);
    assert_ne!(o.tree.height, info.tree.height);
    assert_relative_eq!(
        o.tree.height,
        o.tree.all_branch_lengths().iter().sum::<f64>(),
        epsilon = 1e-4
    );
}

#[test]
fn branch_optimiser_likelihood_increase() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(
        DNAModelType::GTR,
        &[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let cost = SubstitutionLikelihoodCost { model: &model };
    let o = BranchOptimiser::new(&cost, &info).run().unwrap();
    assert!(o.final_logl > o.initial_logl);
    assert_ne!(o.tree.height, info.tree.height);
    assert_relative_eq!(
        o.tree.height,
        o.tree.all_branch_lengths().iter().sum::<f64>(),
        epsilon = 1e-4
    );
}

#[test]
fn branch_optimiser_against_phyml() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(DNAModelType::JC69, &[]).unwrap();
    let cost = SubstitutionLikelihoodCost { model: &model };
    let o = BranchOptimiser::new(&cost, &info).run().unwrap();
    assert!(o.final_logl > o.initial_logl);
    assert_ne!(o.tree.height, info.tree.height);
    assert_relative_eq!(o.final_logl, -4086.56102, epsilon = 1e-4);
    let phyml_tree = from_newick_string("((Gorilla:0.06683711,(Orangutan:0.21859880,Gibbon:0.31145586):0.06570906):0.03853171,Human:0.05356244,Chimpanzee:0.05417982);").unwrap().pop().unwrap();
    for node in o.tree.leaves() {
        let phyml_node = phyml_tree.node(&phyml_tree.idx(&node.id).unwrap());
        assert_relative_eq!(node.blen, phyml_node.blen, epsilon = 1e-4);
    }
    // the only branch that matches after rooting
    assert_relative_eq!(o.tree.node(&Int(2)).blen, 0.03853171, epsilon = 1e-4);
    assert_relative_eq!(o.tree.height, phyml_tree.height, epsilon = 1e-4);
}
