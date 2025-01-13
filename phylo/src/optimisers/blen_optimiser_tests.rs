use std::path::Path;

use approx::assert_relative_eq;

use crate::evolutionary_models::EvoModel;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::BranchOptimiser;
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{dna_models::*, SubstModel, SubstitutionCostBuilder as SCB};
use crate::tree::tree_parser::from_newick;

use crate::tree;

#[test]
fn branch_opt_likelihood_increase_pip() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = PIPModel::<GTR>::new(
        &[0.25, 0.25, 0.25, 0.25],
        &[14.142_1, 0.1414, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let c = PIPCB::new(model.clone(), info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), -5664.780425445042);
    let o = BranchOptimiser::new(c.clone()).run().unwrap();

    assert!(o.final_logl > o.initial_logl);
    assert_eq!(c.cost(), o.initial_logl);

    let new_info = o.cost.info.clone();

    assert_ne!(new_info.tree.height, info.tree.height);
    assert_relative_eq!(
        new_info.tree.height,
        new_info.tree.iter().map(|n| n.blen).sum(),
        epsilon = 1e-4
    );

    let c = PIPCB::new(model, new_info).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_logl);
    assert_eq!(c.cost(), o.final_logl);
}

#[test]
fn branch_opt_likelihood_increase_gtr() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let gtr =
        SubstModel::<GTR>::new(&[0.25, 0.25, 0.25, 0.25], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let o = BranchOptimiser::new(SCB::new(gtr.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();

    assert!(o.final_logl > o.initial_logl);
    assert_ne!(o.cost.tree().height, info.tree.height);

    let c = SCB::new(gtr, o.cost.info.clone()).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_logl);
    assert_eq!(c.cost(), o.final_logl);
}

#[test]
fn branch_optimiser_against_phyml() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let o = BranchOptimiser::new(SCB::new(model.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();
    assert!(o.final_logl > o.initial_logl);

    let result_tree = o.cost.tree();
    assert_ne!(result_tree.height, info.tree.height);
    assert_relative_eq!(o.final_logl, -4086.56102, epsilon = 1e-4);
    let phyml_tree = tree!("((Gorilla:0.06683711,(Orangutan:0.21859880,Gibbon:0.31145586):0.06570906):0.03853171,Human:0.05356244,Chimpanzee:0.05417982);");
    for node in result_tree.leaves() {
        let phyml_node = phyml_tree.node(&phyml_tree.idx(&node.id));
        assert_relative_eq!(node.blen, phyml_node.blen, epsilon = 1e-4);
    }

    assert_eq!(result_tree.robinson_foulds(&info.tree), 0);
    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            result_tree.node(&result_tree.idx(taxon)).blen,
            phyml_tree.node(&phyml_tree.idx(taxon)).blen,
            epsilon = 1e-4
        );
    }
    assert_relative_eq!(result_tree.height, phyml_tree.height, epsilon = 1e-4);

    let c = SCB::new(model, o.cost.info.clone()).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_logl);
    assert_eq!(c.cost(), o.final_logl);
}
