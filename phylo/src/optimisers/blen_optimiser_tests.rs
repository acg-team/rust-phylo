use std::path::Path;

use approx::assert_relative_eq;

use crate::likelihood::TreeSearchCost;
use crate::optimisers::BranchOptimiser;
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{dna_models::*, SubstModel, SubstitutionCostBuilder as SCB, WAG};
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
    );
    let c = PIPCB::new(model.clone(), info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), -5664.780425829528, epsilon = 1e-6);
    let o = BranchOptimiser::new(c.clone()).run().unwrap();

    assert!(o.final_cost > o.initial_cost);
    assert_eq!(c.cost(), o.initial_cost);

    let new_info = o.cost.info.clone();

    assert_ne!(new_info.tree.height, info.tree.height);
    assert_relative_eq!(
        new_info.tree.height,
        new_info.tree.iter().map(|n| n.blen).sum(),
        epsilon = 1e-4
    );

    let c = PIPCB::new(model, new_info).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_eq!(c.cost(), o.final_cost);
}

#[test]
fn branch_opt_likelihood_increase_gtr() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let gtr = SubstModel::<GTR>::new(&[0.25, 0.25, 0.25, 0.25], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let o = BranchOptimiser::new(SCB::new(gtr.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();

    assert!(o.final_cost > o.initial_cost);
    assert_ne!(o.cost.tree().height, info.tree.height);

    let c = SCB::new(gtr, o.cost.info.clone()).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_eq!(c.cost(), o.final_cost);
}

#[test]
fn branch_optimiser_against_phyml() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<JC69>::new(&[], &[]);
    let o = BranchOptimiser::new(SCB::new(model.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();
    assert!(o.final_cost > o.initial_cost);

    let result_tree = o.cost.tree();
    assert_ne!(result_tree.height, info.tree.height);
    assert_relative_eq!(o.final_cost, -4086.56102, epsilon = 1e-4);
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
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_eq!(c.cost(), o.final_cost);
}

#[test]
fn repeated_optimisation_limit() {
    // This used to create -Inf likelihoods due to too long branch lengths and the probability
    // turning to 0.0.
    // This is supposed to run and not crash, no other conditions.

    let fldr = Path::new("./data/");
    let seq_file = fldr.join("p105.msa.fa");
    let info = PIB::new(seq_file).build().unwrap();

    let model = PIPModel::<WAG>::new(&[], &[]);

    let mut cost = PIPCB::new(model, info).build().unwrap();
    let mut prev_cost = f64::NEG_INFINITY;
    let mut final_cost = TreeSearchCost::cost(&cost);
    let max_iterations = 100;
    let epsilon = 1e-5;

    let mut iterations = 0;
    while final_cost - prev_cost > epsilon && iterations < max_iterations {
        iterations += 1;
        prev_cost = final_cost;
        let branch_o = BranchOptimiser::new(cost.clone()).run().unwrap();
        assert!(branch_o.final_cost > branch_o.initial_cost);
        final_cost = branch_o.final_cost;
        cost = branch_o.cost;
    }
}
