use std::path::Path;

use phylo::evolutionary_models::EvoModel;
use phylo::io::write_newick_to_file;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::TopologyOptimiser;
use phylo::phylo_info::PhyloInfoBuilder as PIB;
use phylo::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
use phylo::substitution_models::protein_models::WAG;

fn main() {
    let fldr = Path::new("data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let tree_file = Path::new("./").join("pip_wag_benchmark.newick");

    let model = PIPModel::<WAG>::new(&[], &[0.1, 2.0]).unwrap();
    let info = PIB::new(seq_file.clone()).build().unwrap();
    let c = PIPCB::new(model.clone(), info).build().unwrap();
    let unopt_logl = c.cost();

    let o = TopologyOptimiser::new(c).run().unwrap();
    assert!(o.final_logl >= unopt_logl);
    assert_eq!(o.initial_logl, unopt_logl);
    assert_eq!(o.final_logl, o.cost.cost());
    assert!(write_newick_to_file(&[o.cost.tree().clone()], tree_file.clone()).is_ok());
    println!("Run done");
}
