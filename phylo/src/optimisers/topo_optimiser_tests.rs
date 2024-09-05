use std::path::PathBuf;

use approx::assert_relative_eq;
use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::evolutionary_models::ProteinModelType;
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::PhyloOptimiser;
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::PIPModel;
use crate::substitution_models::{DNASubstModel, ProteinSubstModel};
use crate::tree::tree_parser::from_newick_string;
use crate::{
    evolutionary_models::{DNAModelType::*, EvoModel},
    optimisers::TopologyOptimiser,
};

#[test]
fn primate_topology_from_right_tree() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = model.cost(&info);
    let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    assert!(o.final_logl >= unopt_logl);
    // Likelihood from PhyML
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-6);

    let opt_tree = &o.i.tree;
    let phyml_tree = from_newick_string(&String::from(
        "((Gorilla:0.05864183,(Orangutan:0.21100967,Gibbon:0.27996761)0.999800:0.09804101)0.986100:0.03441881,Human:0.05073325,Chimpanzee:0.06040307);",
    ))
    .unwrap()
    .pop()
    .unwrap();
    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            opt_tree.node(&opt_tree.idx(taxon)).blen,
            phyml_tree.node(&phyml_tree.idx(taxon)).blen,
            epsilon = 1e-5
        );
    }
    assert_relative_eq!(opt_tree.height, phyml_tree.height, epsilon = 1e-5);
}

#[test]
fn primate_topology_from_wrong_tree() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/wrong_tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(JC69, &[]).unwrap();
    let unopt_logl = model.cost(&info);
    let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    assert!(o.final_logl >= unopt_logl);
    // Likelihood from PhyML
    assert_relative_eq!(o.final_logl, -4038.721121221992, epsilon = 1e-5);

    let opt_tree = &o.i.tree;
    let phyml_tree = from_newick_string(&String::from(
        "((Gorilla:0.05864183,(Orangutan:0.21100967,Gibbon:0.27996761)0.999800:0.09804101)0.986100:0.03441881,Human:0.05073325,Chimpanzee:0.06040307);",
    ))
    .unwrap()
    .pop()
    .unwrap();
    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            opt_tree.node(&opt_tree.idx(taxon)).blen,
            phyml_tree.node(&phyml_tree.idx(taxon)).blen,
            epsilon = 1e-3
        );
    }
    // Different branch length due to different rooting, adding up to the same height
    let human = opt_tree.node(&opt_tree.idx("Human"));
    assert_relative_eq!(
        opt_tree.node(&opt_tree.idx("Chimpanzee")).blen
            + opt_tree.node(&human.parent.unwrap()).blen,
        phyml_tree.node(&phyml_tree.idx("Chimpanzee")).blen,
        epsilon = 1e-5
    );
    assert_relative_eq!(opt_tree.height, phyml_tree.height, epsilon = 1e-5);
}

#[test]
fn simple_topology_optimisation() {
    let tree = from_newick_string(&String::from(
        "(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);",
    ))
    .unwrap()
    .pop()
    .unwrap();
    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
        Record::with_attrs("D", None, b"TTATATATAT"),
    ]);
    let info = PhyloInfoBuilder::build_from_objects(sequences, tree).unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = model.cost(&info);
    let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    assert!(o.final_logl >= unopt_logl);
}

#[test]
fn protein_topology_optimisation_good_start() {
    // let info = PhyloInfoBuilder::with_attrs(
    //     PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"),
    //     PathBuf::from("./data/phyml_protein_example/tree.newick"),
    // )
    // .build()
    // .unwrap();
    // let model = ProteinSubstModel::new(ProteinModelType::WAG, &[]).unwrap();
    // let unopt_logl = model.cost(&info);
    // let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    // assert!(o.final_logl >= unopt_logl);
    // The above ran in 888.89s

    // The optimisation itself takes too long, the tree checked here is the output of the above
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/phyml_result.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(ProteinModelType::WAG, &[]).unwrap();
    let logl = model.cost(&info);

    // compare the tree height and logl to the output of PhyML
    assert_relative_eq!(info.tree.height, 1.05242, epsilon = 1e-2);
    assert_relative_eq!(logl, -4490.78548891, epsilon = 1e-3);
}

#[test]
fn protein_topology_optimisation_nj_start() {
    // let info = PhyloInfoBuilder::new(PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"))
    //     .build()
    //     .unwrap();
    // let model = ProteinSubstModel::new(ProteinModelType::WAG, &[]).unwrap();
    // let unopt_logl = model.cost(&info);
    // let o = TopologyOptimiser::new(&model, &info).run().unwrap();
    // assert!(o.final_logl >= unopt_logl);
    // The above ran in 493.88s

    // The optimisation itself takes too long, the tree checked here is the output of the above
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_example/nogap_seqs.fasta"),
        PathBuf::from("./data/phyml_protein_example/optimisation_nj_start.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(ProteinModelType::WAG, &[]).unwrap();
    let logl = model.cost(&info);

    // compare the tree height and logl to the output of PhyML
    assert_relative_eq!(info.tree.height, 1.05242, epsilon = 1e-2);
    assert_relative_eq!(logl, -4490.78548891, epsilon = 1e0);
}

#[test]
fn pip_vs_subst_dna_tree() {
    let info = &PhyloInfoBuilder::new(PathBuf::from("./data/sim/K80/K80.fasta"))
        .build()
        .unwrap();
    let pip = PIPModel::<DNASubstModel>::new(K80, &[4.0, 1.0]).unwrap();
    let initial_logl = pip.cost(info);
    let o_pip = TopologyOptimiser::new(&pip, info).run().unwrap();
    assert!(o_pip.final_logl > initial_logl);
    assert_relative_eq!(o_pip.initial_logl, initial_logl);

    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let initial_logl = model.cost(info);
    let o_k80 = TopologyOptimiser::new(&model, info).run().unwrap();
    assert!(o_k80.final_logl > initial_logl);
    assert_relative_eq!(o_k80.initial_logl, initial_logl);
    assert_eq!(o_pip.i.tree.robinson_foulds(&o_k80.i.tree), 0);
}
