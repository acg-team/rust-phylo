use std::path::Path;

use approx::assert_relative_eq;

use crate::evolutionary_models::{DNAModelType, EvoModel};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{BranchOptimiser, PhyloOptimiser};
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::PIPDNAModel;
use crate::substitution_models::DNASubstModel;
use crate::tree::{tree_parser::from_newick, NodeIdx::Internal as Int};

use crate::tree;

#[test]
fn likelihood_increase_pip() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = PIPDNAModel::new(
        DNAModelType::GTR,
        &[
            14.142_1, 0.1414, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    assert_relative_eq!(model.cost(&info, false), -5664.780425445042);
    let o = BranchOptimiser::new(&model, &info).run().unwrap();
    assert_relative_eq!(model.cost(&info, true), o.initial_logl);
    assert!(o.final_logl > o.initial_logl);
    assert_ne!(o.i.tree.height, info.tree.height);
    assert_relative_eq!(
        o.i.tree.height,
        o.i.tree.iter().map(|n| n.blen).sum(),
        epsilon = 1e-4
    );
    assert_relative_eq!(model.cost(&o.i, true), o.final_logl);
}

#[test]
fn branch_optimiser_likelihood_increase() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = DNASubstModel::new(
        DNAModelType::GTR,
        &[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let o = BranchOptimiser::new(&model, &info).run().unwrap();
    assert!(o.final_logl > o.initial_logl);
    assert_ne!(o.i.tree.height, info.tree.height);
    assert_relative_eq!(
        o.i.tree.height,
        o.i.tree.iter().map(|n| n.blen).sum(),
        epsilon = 1e-4
    );
}

#[test]
fn branch_optimiser_against_phyml() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = DNASubstModel::new(DNAModelType::JC69, &[]).unwrap();
    let o = BranchOptimiser::new(&model, &info).run().unwrap();
    assert!(o.final_logl > o.initial_logl);
    assert_ne!(o.i.tree.height, info.tree.height);
    assert_relative_eq!(o.final_logl, -4086.56102, epsilon = 1e-4);
    let phyml_tree = tree!("((Gorilla:0.06683711,(Orangutan:0.21859880,Gibbon:0.31145586):0.06570906):0.03853171,Human:0.05356244,Chimpanzee:0.05417982);");
    for node in o.i.tree.leaves() {
        let phyml_node = phyml_tree.node(&phyml_tree.idx(&node.id));
        assert_relative_eq!(node.blen, phyml_node.blen, epsilon = 1e-4);
    }
    // the only branch that matches after rooting
    assert_relative_eq!(o.i.tree.node(&Int(2)).blen, 0.03853171, epsilon = 1e-4);
    assert_relative_eq!(o.i.tree.height, phyml_tree.height, epsilon = 1e-4);
}
