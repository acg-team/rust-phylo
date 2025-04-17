use std::path::Path;

// use approx::assert_relative_eq;

use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::io::{read_newick_from_file, write_newick_to_file};
use phylo::likelihood::{ModelSearchCost, TreeSearchCost};
use phylo::optimisers::{ModelOptimiser, TopologyOptimiser};
use phylo::phylo_info::PhyloInfoBuilder;
// use phylo::pip_model::{PIPCostBuilder, PIPModel};
use phylo::substitution_models::protein_models::WAG;

use ftail::Ftail;
use log::LevelFilter;
use phylo::substitution_models::{SubstModel, SubstitutionCostBuilder};

fn main() {
    let _ = Ftail::new()
        .console(LevelFilter::Info)
        .single_file("logs", true, LevelFilter::Debug)
        .init();

    let fldr = Path::new("data/demo/indelible_data");
    let seq_file = fldr.join("323054_indelible_TRUE.fas");
    let tree_file = fldr.join("323054.nwk");
    let true_tree = read_newick_from_file(&tree_file).unwrap().pop().unwrap();

    // Lambda:402.169
    // Mu:0.481267

    let model = SubstModel::<WAG>::new(&[], &[]);
    let info = PhyloInfoBuilder::new(seq_file).build().unwrap();
    let c = SubstitutionCostBuilder::new(model.clone(), info)
        .build()
        .unwrap();
    let unopt_logl = ModelSearchCost::cost(&c);

    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, TreeSearchCost::cost(&o.cost));

    let o = TopologyOptimiser::new(o.cost).run().unwrap();
    assert!(write_newick_to_file(
        &[TreeSearchCost::tree(&o.cost).clone()],
        fldr.join("323054_JATI_tree_subst.nwk")
    )
    .is_ok());
    assert_eq!(o.cost.tree().robinson_foulds(&true_tree), 0);
    println!("Run done");
}
